from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, validator, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, date
import logging
import json
import os
from ml_anomaly_detector import EnsembleAnomalyDetector
import random
from twilio.rest import Client
from dotenv import load_dotenv
import asyncio
from enum import Enum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Expense Management API with ML Anomaly Detection",
    description="A comprehensive expense management system with ML-powered anomaly detection and policy enforcement",
    version="2.1.0",  # Updated version
    docs_url="/docs",
    redoc_url="/redoc"
)

# Create directories for models and data
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Define possible values
employees = ['E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110']
departments = ['Finance', 'Engineering', 'Sales', 'HR', 'Marketing', 'Operations']
purposes = ['Travel', 'Meals', 'Supplies', 'Client Entertainment', 'Training', 'Software Subscription']

# Create 200 synthetic expense records
expenses_data = {
    'expense_id': list(range(1, 201)),
    'employee_id': [random.choice(employees) for _ in range(200)],
    'department': [random.choice(departments) for _ in range(200)],
    'amount': [round(random.uniform(50, 2500), 2) for _ in range(200)],  # expense between $50–$2500
    'purpose': [random.choice(purposes) for _ in range(200)],
    'status': [random.choice(['approved', 'pending', 'rejected']) for _ in range(200)],
    'submission_date': [datetime.now() for _ in range(200)],
    'anomaly_confidence': [round(random.uniform(0.0, 1.0), 2) for _ in range(200)],
    'ml_anomaly': [random.choice([False, False, False, True]) for _ in range(200)]  # ~25% anomalies
}

departments_data = {
    'department': ['Sales', 'Engineering', 'HR', 'Finance'],
    'monthly_budget': [31500, 37500, 12500, 33500],
    'auto_approve_limit': [225.75, 246.97, 128.81, 280.34],
    'escalation_limit': [655.14, 673.51, 433.31, 826.38],
    'budget_usage': [28444.48, 34081.32, 11335.48, 30276.76],
    'budget_reset_date': [date.today().replace(day=1)] * 4,
    'manager_phone': ['+917879287098', '+917879287098', '+917879287098', '+917879287098']  # Manager phone numbers
}

expenses_df = pd.DataFrame(expenses_data)
departments_df = pd.DataFrame(departments_data)

# Initialize ML detector
ml_detector = EnsembleAnomalyDetector()

# Audit trail for learning and governance
audit_trail = []

# Twilio Configuration
class AlertType(Enum):
    ANOMALY_DETECTED = "anomaly_detected"
    BUDGET_EXCEEDED = "budget_exceeded"
    ESCALATION_REQUIRED = "escalation_required"
    CFO_REPORT_READY = "cfo_report_ready"
    EXPENSE_APPROVED = "expense_approved"
    EXPENSE_REJECTED = "expense_rejected"

class MessagingService:
    """Service for sending alerts and notifications via Twilio"""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.whatsapp_from = os.getenv("TWILIO_WHATSAPP_FROM")
        self.sms_from = os.getenv("TWILIO_SMS_FROM")
        self.content_sid = os.getenv("CONTENT_SID")
        self.client = None
        self.is_configured = False
        
        if all([self.account_sid, self.auth_token]):
            try:
                self.client = Client(self.account_sid, self.auth_token)
                self.is_configured = True
                logger.info("Twilio messaging service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
        else:
            logger.warning("Twilio credentials not found. Messaging service will run in simulation mode.")
    
    async def send_alert(self, 
                        alert_type: AlertType, 
                        recipient_phone: str, 
                        expense_data: Optional[Dict] = None,
                        department_data: Optional[Dict] = None,
                        report_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Send alert message based on alert type"""
        
        if not self.is_configured:
            logger.warning(f"SIMULATION: Would send {alert_type.value} alert to {recipient_phone}")
            return {"status": "simulated", "message": "Messaging service not configured"}
        
        try:
            # Determine message content based on alert type
            message_data = self._prepare_message_content(
                alert_type, 
                expense_data or {},  # Provide empty dict if None
                department_data or {},  # Provide empty dict if None
                report_data or {}  # Provide empty dict if None
            )
            
            # Send message
            if self.whatsapp_from and "whatsapp:" in self.whatsapp_from:
                # Send WhatsApp message
                message = self.client.messages.create(
                    from_=self.whatsapp_from,
                    body=message_data["body"],
                    to=f"whatsapp:{recipient_phone}"
                )
            else:
                # Send SMS
                message = self.client.messages.create(
                    body=message_data["body"],
                    from_=self.sms_from or self.whatsapp_from,
                    to=recipient_phone
                )
            
            # Log the alert
            alert_record = {
                "timestamp": datetime.now(),
                "alert_type": alert_type.value,
                "recipient": recipient_phone,
                "message_sid": message.sid,
                "status": message.status,
                "content": message_data["body"][:100] + "..." if len(message_data["body"]) > 100 else message_data["body"]
            }
            audit_trail.append(alert_record)
            
            logger.info(f"Alert sent: {alert_type.value} to {recipient_phone} (SID: {message.sid})")
            
            return {
                "status": "sent",
                "message_sid": message.sid,
                "status_code": message.status,
                "recipient": recipient_phone,
                "alert_type": alert_type.value
            }
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "alert_type": alert_type.value
            }
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "alert_type": alert_type.value
            }
    
    def _prepare_message_content(self, alert_type: AlertType, expense_data: Optional[Dict], 
                                department_data: Optional[Dict], report_data: Optional[Dict]) -> Dict[str, str]:
        """Prepare message content based on alert type"""
        
        templates = {
            AlertType.ANOMALY_DETECTED: {
                "body": f"""🚨 EXPENSE ANOMALY DETECTED
Expense ID: {expense_data.get('expense_id', 'N/A')}
Employee: {expense_data.get('employee_id', 'N/A')}
Amount: ${expense_data.get('amount', 0):.2f}
Purpose: {expense_data.get('purpose', 'N/A')}
Confidence: {expense_data.get('confidence_score', 0):.1%}
Reason: {expense_data.get('reasoning', 'Suspicious pattern detected')}
Action Required: Please review at your earliest convenience."""
            },
            AlertType.BUDGET_EXCEEDED: {
                "body": f"""⚠️ BUDGET ALERT: {department_data.get('department', 'Department')}
Current Usage: ${department_data.get('budget_usage', 0):.2f} / ${department_data.get('monthly_budget', 0):.2f}
Utilization: {(department_data.get('budget_usage', 0) / department_data.get('monthly_budget', 1) * 100):.1f}%
Status: {"EXCEEDED" if department_data.get('budget_usage', 0) > department_data.get('monthly_budget', 0) else "CRITICAL"}
Action: Review department expenses and consider budget adjustment."""
            },
            AlertType.ESCALATION_REQUIRED: {
                "body": f"""🔔 ESCALATION REQUIRED
Expense ID: {expense_data.get('expense_id', 'N/A')}
Amount: ${expense_data.get('amount', 0):.2f} (Limit: ${department_data.get('escalation_limit', 0):.2f})
Employee: {expense_data.get('employee_id', 'N/A')}
Department: {expense_data.get('department', 'N/A')}
Action: Senior approval required. Please review in the system."""
            },
            AlertType.CFO_REPORT_READY: {
                "body": f"""📊 CFO REPORT ALERT
Monthly expense report is ready.
Total Expenses: {report_data.get('total_expenses_processed', 0)}
Total Amount: ${report_data.get('total_amount_processed', 0):.2f}
Anomaly Rate: {report_data.get('ml_insights', {}).get('anomaly_rate', 0):.1f}%
Risk Level: {report_data.get('risk_assessment', {}).get('risk_level', 'N/A')}
Access: /cfo/report endpoint"""
            },
            AlertType.EXPENSE_APPROVED: {
                "body": f"""✅ EXPENSE APPROVED
Expense ID: {expense_data.get('expense_id', 'N/A')}
Amount: ${expense_data.get('amount', 0):.2f}
Approved By: {expense_data.get('reviewer', 'System')}
Status: Approved and processed."""
            },
            AlertType.EXPENSE_REJECTED: {
                "body": f"""❌ EXPENSE REJECTED
Expense ID: {expense_data.get('expense_id', 'N/A')}
Amount: ${expense_data.get('amount', 0):.2f}
Rejected By: {expense_data.get('reviewer', 'System')}
Reason: {expense_data.get('comments', 'Policy violation')}"""
            }
        }
        
        return templates.get(alert_type, {"body": "Alert: Action required in expense management system."})

# Initialize messaging service
messaging_service = MessagingService()

# Pydantic Models
class ExpenseSubmission(BaseModel):
    employee_id: str = Field(..., pattern=r'^E\d{3}$', description="Employee ID in format EXXX")
    department: str = Field(..., description="Department name")
    amount: float = Field(..., gt=0, description="Expense amount must be positive")
    purpose: str = Field(..., description="Purpose of expense")
    
    @validator('department')
    def department_exists(cls, v):
        valid_departments = ['Sales', 'Engineering', 'HR', 'Finance']
        if v not in valid_departments:
            raise ValueError(f"Department must be one of {valid_departments}")
        return v

class ExpenseResponse(BaseModel):
    expense_id: int
    employee_id: str
    department: str
    amount: float
    purpose: str
    status: str
    reasoning: str
    confidence_score: Optional[float] = None
    needs_human_review: bool
    anomaly_details: Optional[Dict[str, Any]] = None
    alert_sent: Optional[bool] = False
    alert_details: Optional[Dict[str, Any]] = None

class DepartmentUpdate(BaseModel):
    monthly_budget: Optional[float] = Field(None, gt=0)
    auto_approve_limit: Optional[float] = Field(None, gt=0)
    escalation_limit: Optional[float] = Field(None, gt=0)
    manager_phone: Optional[str] = Field(None, pattern=r'^\+\d{10,15}$', description="Manager phone number in E.164 format")

class ApprovalDecision(BaseModel):
    approved: bool
    reviewer: str = Field(..., min_length=2)
    comments: Optional[str] = Field(None, max_length=500)
    send_notification: Optional[bool] = Field(True, description="Send notification to employee")

class MLTrainingResponse(BaseModel):
    status: str
    training_samples: Optional[int] = None
    anomalies_detected: Optional[int] = None
    error: Optional[str] = None

class AlertRequest(BaseModel):
    alert_type: AlertType
    recipient_phone: str = Field(..., pattern=r'^\+\d{10,15}$')
    expense_id: Optional[int] = None
    department: Optional[str] = None
    custom_message: Optional[str] = Field(None, max_length=500)

# Enhanced Policy Check Service with Alert Integration
class PolicyChecker:
    @staticmethod
    async def check_policy(expense: ExpenseSubmission, department_data: pd.Series) -> Dict[str, Any]:
        """Check expense against department policies with alert triggers"""
        reasoning = []
        needs_human_review = False
        confidence_score = 1.0
        alerts = []
        
        # Auto-approval check
        if expense.amount <= department_data['auto_approve_limit']:
            reasoning.append(f"Amount (${expense.amount:.2f}) within auto-approval limit (${department_data['auto_approve_limit']:.2f})")
            status = "auto_approved"
        # Escalation check
        elif expense.amount > department_data['escalation_limit']:
            reasoning.append(f"Amount (${expense.amount:.2f}) exceeds escalation limit (${department_data['escalation_limit']:.2f}) - requires senior approval")
            status = "escalated"
            needs_human_review = True
            confidence_score = 0.7
            alerts.append(AlertType.ESCALATION_REQUIRED)
        # Normal approval
        else:
            reasoning.append(f"Amount (${expense.amount:.2f}) requires standard approval (limit: ${department_data['auto_approve_limit']:.2f})")
            status = "pending"
            needs_human_review = True
            confidence_score = 0.9
        
        # Budget check
        remaining_budget = department_data['monthly_budget'] - department_data['budget_usage']
        if expense.amount > remaining_budget:
            reasoning.append(f"Warning: Expense exceeds remaining budget (${remaining_budget:.2f})")
            status = "budget_exceeded"
            needs_human_review = True
            confidence_score = 0.5
            alerts.append(AlertType.BUDGET_EXCEEDED)
        elif expense.amount > remaining_budget * 0.8:
            reasoning.append(f"Warning: Expense uses more than 80% of remaining budget (${remaining_budget:.2f})")
            confidence_score *= 0.8
            if remaining_budget * 0.1 < expense.amount:  # If using >10% of remaining budget
                alerts.append(AlertType.BUDGET_EXCEEDED)
        
        # Purpose-based checks
        if expense.purpose == "Client Entertainment" and expense.amount > 1000:
            reasoning.append("High amount for Client Entertainment - requires documentation")
            confidence_score *= 0.8
        
        return {
            "status": status,
            "reasoning": "; ".join(reasoning),
            "needs_human_review": needs_human_review,
            "confidence_score": confidence_score,
            "alerts": alerts
        }

# Enhanced Notification Service with Twilio Integration
class NotificationService:
    @staticmethod
    async def notify_approver(expense_id: int, employee_id: str, department: str, amount: float, 
                             purpose: str, reasoning: str, confidence_score: float, 
                             anomaly_details: Dict, alert_type: AlertType = AlertType.ANOMALY_DETECTED):
        """Notify approvers via multiple channels"""
        
        try:
            # Get manager phone number from department data
            dept_data = departments_df[departments_df['department'] == department]
            
            # Check if department data exists and has manager_phone
            if not dept_data.empty and 'manager_phone' in dept_data.columns:
                manager_phone = dept_data.iloc[0]['manager_phone']
                
                # Prepare expense data for alert
                expense_data = {
                    'expense_id': expense_id,
                    'employee_id': employee_id,
                    'department': department,
                    'amount': amount,
                    'purpose': purpose,
                    'reasoning': reasoning,
                    'confidence_score': confidence_score
                }
                
                # Send alert via Twilio if manager phone exists
                if manager_phone and pd.notna(manager_phone):
                    await messaging_service.send_alert(
                        alert_type=alert_type,
                        recipient_phone=manager_phone,
                        expense_data=expense_data,
                        department_data=dept_data.iloc[0].to_dict() if not dept_data.empty else None
                    )
                else:
                    logger.warning(f"No valid manager phone for department {department}")
            else:
                logger.warning(f"Department {department} not found or no manager phone configured")
        
        except Exception as e:
            logger.error(f"Error in notify_approver: {e}")
        
        # Also log to console for backup (this part is working)
        message = f"""
        EXPENSE REVIEW REQUIRED
        ======================
        Expense ID: {expense_id}
        Employee: {employee_id}
        Department: {department}
        Amount: ${amount:.2f}
        Purpose: {purpose}
        
        Policy Reasoning: {reasoning}
        Anomaly Confidence: {confidence_score:.2f}
        
        Anomaly Details:
        {json.dumps(anomaly_details, indent=2)}
        
        Please review and approve/reject this expense.
        """
        logger.info(f"APPROVAL NOTIFICATION:\n{message}")
    
    @staticmethod
    async def notify_cfo(report_data: Dict):
        """Notify CFO of significant events"""
        # Send alert to CFO
        cfo_phone = os.getenv("CFO_PHONE_NUMBER")
        if cfo_phone:
            await messaging_service.send_alert(
                alert_type=AlertType.CFO_REPORT_READY,
                recipient_phone=cfo_phone,
                report_data=report_data
            )
        
        # Also log to console
        message = f"""
        CFO ALERT: Monthly Expense Report
        ================================
        Total Expenses: {report_data.get('total_expenses_processed', 0)}
        Total Amount: ${report_data.get('total_amount_processed', 0):.2f}
        Anomaly Rate: {report_data.get('ml_insights', {}).get('anomaly_rate', 0):.1f}%
        
        High Risk Departments:
        {json.dumps(report_data.get('ml_insights', {}).get('high_risk_departments', {}), indent=2)}
        """
        logger.info(f"CFO NOTIFICATION:\n{message}")

# CFO Reporting Service remains the same
class CFOReportService:
    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy/pandas types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: CFOReportService._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [CFOReportService._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return str(obj)  # Convert tuples to strings
        else:
            return obj

    @staticmethod
    def _flatten_multiindex_columns(df_dict):
        """Flatten multi-level column indexes from pandas DataFrames"""
        flattened = {}
        for key, value in df_dict.items():
            if isinstance(key, tuple):
                # Convert tuple key to string
                new_key = "_".join(str(k) for k in key if k)
                flattened[new_key] = CFOReportService._convert_to_serializable(value)
            else:
                flattened[str(key)] = CFOReportService._convert_to_serializable(value)
        return flattened

    @staticmethod
    def generate_report() -> Dict[str, Any]:
        """Generate CFO summary report with ML insights"""
        # Expense trends by department - create a simpler structure
        dept_summary = expenses_df.groupby('department').agg({
            'amount': ['sum', 'mean', 'count'],
            'employee_id': 'nunique'
        }).round(2)
        
        # Convert to a simpler dictionary structure
        department_summary_simple = {}
        for dept in dept_summary.index:
            dept_data = {}
            for col in dept_summary.columns:
                # Flatten column names
                if isinstance(col, tuple):
                    col_name = f"{col[0]}_{col[1]}"
                else:
                    col_name = col
                dept_data[col_name] = CFOReportService._convert_to_serializable(dept_summary.loc[dept, col])
            department_summary_simple[dept] = dept_data
        
        # Budget utilization
        budget_utilization = []
        for dept in departments_df['department']:
            dept_data = departments_df[departments_df['department'] == dept].iloc[0]
            expenses_data = expenses_df[expenses_df['department'] == dept]
            utilization = (dept_data['budget_usage'] / dept_data['monthly_budget']) * 100
            
            budget_utilization.append({
                'department': dept,
                'budget_used': float(dept_data['budget_usage']),
                'total_budget': float(dept_data['monthly_budget']),
                'utilization_percent': round(utilization, 2),
                'remaining_budget': float(dept_data['monthly_budget'] - dept_data['budget_usage']),
                'avg_expense_amount': round(float(expenses_data['amount'].mean()), 2) if len(expenses_data) > 0 else 0,
                'expense_count': int(len(expenses_data))
            })
        
        # ML insights
        ml_insights = CFOReportService._get_ml_insights()
        
        # Approval statistics
        status_counts = expenses_df['status'].value_counts().to_dict()
        
        return {
            'generated_at': datetime.now().isoformat(),
            'report_period': 'current_month',
            'budget_utilization': budget_utilization,
            'department_summary': department_summary_simple,
            'ml_insights': ml_insights,
            'approval_statistics': status_counts,
            'total_expenses_processed': int(len(expenses_df)),
            'total_amount_processed': float(expenses_df['amount'].sum()),
            'risk_assessment': CFOReportService._get_risk_assessment()
        }
    
    @staticmethod
    def _get_ml_insights() -> Dict[str, Any]:
        """Get ML model insights and performance"""
        if len(expenses_df) == 0:
            return {}
            
        # Calculate anomaly statistics
        anomaly_stats = {
            'total_anomalies_detected': int(expenses_df['ml_anomaly'].sum()),
            'anomaly_rate': float((expenses_df['ml_anomaly'].sum() / len(expenses_df)) * 100),
            'avg_anomaly_confidence': float(expenses_df['anomaly_confidence'].mean()),
            'high_confidence_anomalies': int(len(expenses_df[expenses_df['anomaly_confidence'] > 0.7]))
        }
        
        # Department risk analysis
        dept_risk = expenses_df.groupby('department').agg({
            'ml_anomaly': 'mean',
            'anomaly_confidence': 'mean',
            'amount': 'sum'
        }).round(4)
        
        # Convert to native Python types
        high_risk_departments = {}
        for dept, risk in dept_risk['ml_anomaly'].items():
            high_risk_departments[dept] = float(risk)
        
        anomaly_stats['high_risk_departments'] = high_risk_departments
        
        # Model performance (mock - in production would use actual metrics)
        anomaly_stats['model_performance'] = {
            'estimated_precision': 0.82,
            'estimated_recall': 0.75,
            'estimated_f1_score': 0.78,
            'training_size': int(len(expenses_df)),
            'model_status': ml_detector.ml_detector.is_trained
        }
        
        return anomaly_stats
    
    @staticmethod
    def _get_risk_assessment() -> Dict[str, Any]:
        """Generate risk assessment for CFO"""
        risks = []
        
        # Budget risks
        for dept in departments_df['department']:
            dept_data = departments_df[departments_df['department'] == dept].iloc[0]
            utilization = (dept_data['budget_usage'] / dept_data['monthly_budget']) * 100
            
            if utilization > 90:
                risks.append(f"{dept} department exceeding 90% budget utilization")
            elif utilization > 80:
                risks.append(f"{dept} department approaching budget limit ({utilization:.1f}%)")
        
        # Anomaly risks
        high_risk_expenses = expenses_df[expenses_df['anomaly_confidence'] > 0.7]
        if len(high_risk_expenses) > 5:
            risks.append(f"Multiple high-confidence anomalies detected ({len(high_risk_expenses)})")
        
        # Employee risks
        employee_submissions = expenses_df['employee_id'].value_counts()
        frequent_submitters = employee_submissions[employee_submissions > 5]
        if len(frequent_submitters) > 0:
            risks.append(f"Frequent submitters: {', '.join(frequent_submitters.index)}")
        
        return {
            'risk_level': 'HIGH' if len(risks) > 3 else 'MEDIUM' if len(risks) > 1 else 'LOW',
            'identified_risks': risks,
            'risk_count': len(risks)
        }

# Learning and Governance Service
class LearningGovernanceService:
    @staticmethod
    def record_decision(expense_id: int, action: str, reasoning: str, user: str, 
                       feedback: Optional[str] = None, metadata: Optional[Dict] = None):
        """Record decisions for learning and transparency"""
        audit_record = {
            'timestamp': datetime.now(),
            'expense_id': expense_id,
            'action': action,
            'reasoning': reasoning,
            'user': user,
            'feedback': feedback,
            'metadata': metadata or {}
        }
        audit_trail.append(audit_record)
        logger.info(f"AUDIT: {action} for expense {expense_id} by {user}")
        
    @staticmethod
    def get_audit_stats() -> Dict[str, Any]:
        """Get statistics from audit trail"""
        if not audit_trail:
            return {}
            
        # Use the original audit_trail which has datetime objects
        df = pd.DataFrame(audit_trail)
        
        # Count recent activity using datetime objects
        today = date.today()
        recent_activity = 0
        for record in audit_trail:
            if record['timestamp'].date() == today:
                recent_activity += 1
        
        return {
            'total_actions': len(audit_trail),
            'actions_by_type': df['action'].value_counts().to_dict(),
            'recent_activity': recent_activity
        }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to Expense Management API with ML Anomaly Detection & Alert System",
        "version": "2.1.0",
        "status": "operational",
        "features": {
            "ml_anomaly_detection": True,
            "twilio_alerts": messaging_service.is_configured,
            "cfo_reporting": True,
            "audit_trail": True
        },
        "endpoints": {
            "submit_expense": "POST /expenses/submit",
            "review_expense": "POST /expenses/{id}/review", 
            "cfo_report": "GET /cfo/report",
            "ml_training": "POST /ml/retrain",
            "audit_trail": "GET /audit/trail",
            "send_alert": "POST /alerts/send",
            "messaging_status": "GET /messaging/status"
        }
    }

@app.post("/expenses/submit", response_model=ExpenseResponse)
async def submit_expense(expense: ExpenseSubmission, background_tasks: BackgroundTasks):
    """Submit a new expense for processing with ML anomaly detection and alerts"""

    global expenses_df
    
    # Validate department exists
    if expense.department not in departments_df['department'].values:
        raise HTTPException(status_code=400, detail="Invalid department")
    
    # Generate expense ID
    expense_id = len(expenses_df) + 1
    
    # Get department data
    dept_data = departments_df[departments_df['department'] == expense.department].iloc[0]
    
    # Policy check
    policy_result = await PolicyChecker.check_policy(expense, dept_data)
    
    # ML Anomaly detection
    current_expense_dict = {
        'employee_id': expense.employee_id,
        'department': expense.department,
        'amount': expense.amount,
        'purpose': expense.purpose
    }
    
    anomaly_result = ml_detector.detect(expenses_df, current_expense_dict)
    
    # Combine results
    final_confidence = policy_result['confidence_score'] * (1 - anomaly_result['confidence_score'])
    final_status = policy_result['status']
    final_reasoning = policy_result['reasoning']
    
    anomaly_details = {
        "ml_confidence": anomaly_result["confidence_score"],
        "anomaly_type": anomaly_result["anomaly_type"],
        "is_anomalous": anomaly_result["is_anomalous"]
    }
    
    alert_sent = False
    alert_details = None
    
    if anomaly_result['is_anomalous']:
        final_status = "suspicious"
        anomaly_details.update({
            "ml_details": anomaly_result.get("ml_result", {}),
            "rule_details": anomaly_result.get("rule_result", {})
        })
        
        anomaly_reasons = []
        if 'ml_result' in anomaly_result and 'detected_rules' in anomaly_result['ml_result']:
            anomaly_reasons.extend(anomaly_result['ml_result']['detected_rules'])
        if 'rule_result' in anomaly_result and 'detected_patterns' in anomaly_result['rule_result']:
            anomaly_reasons.extend(anomaly_result['rule_result']['detected_patterns'])
        
        if anomaly_reasons:
            final_reasoning += f"; Anomalies: {', '.join(anomaly_reasons)}"
        
        final_confidence *= 0.5
        
        # Trigger alert for anomaly
        background_tasks.add_task(
            NotificationService.notify_approver,
            expense_id, expense.employee_id, expense.department, expense.amount,
            expense.purpose, final_reasoning, final_confidence, anomaly_details,
            AlertType.ANOMALY_DETECTED
        )
        alert_sent = True
        alert_details = {"type": "anomaly_detected", "timestamp": datetime.now().isoformat()}
    
    # Check if any policy alerts need to be sent
    if policy_result.get('alerts'):
        for alert_type in policy_result['alerts']:
            background_tasks.add_task(
                NotificationService.notify_approver,
                expense_id, expense.employee_id, expense.department, expense.amount,
                expense.purpose, final_reasoning, final_confidence, anomaly_details,
                alert_type
            )
            alert_sent = True
            alert_details = {"type": alert_type.value, "timestamp": datetime.now().isoformat()}
    
    # Add to expenses database
    new_expense = {
        'expense_id': expense_id,
        'employee_id': expense.employee_id,
        'department': expense.department,
        'amount': expense.amount,
        'purpose': expense.purpose,
        'status': final_status,
        'submission_date': datetime.now(),
        'anomaly_confidence': anomaly_result['confidence_score'],
        'ml_anomaly': anomaly_result['is_anomalous'],
        'alert_sent': alert_sent
    }
    
    
    expenses_df = pd.concat([expenses_df, pd.DataFrame([new_expense])], ignore_index=True)
    
    # Retrain ML model periodically (every 20 new expenses)
    if len(expenses_df) % 20 == 0:
        background_tasks.add_task(retrain_ml_model_background)
    
    # Update department budget usage if auto-approved
    if final_status == "auto_approved":
        departments_df.loc[departments_df['department'] == expense.department, 'budget_usage'] += expense.amount
    
    # Trigger human review if needed
    needs_review = policy_result['needs_human_review'] or anomaly_result['is_anomalous']
    
    # Record in audit trail
    LearningGovernanceService.record_decision(
        expense_id, "submission", final_reasoning, "system",
        f"ML_confidence: {anomaly_result['confidence_score']:.3f}, Alert_sent: {alert_sent}",
        {"anomaly_result": anomaly_result, "alert_sent": alert_sent}
    )
    
    return ExpenseResponse(
        expense_id=expense_id,
        employee_id=expense.employee_id,
        department=expense.department,
        amount=expense.amount,
        purpose=expense.purpose,
        status=final_status,
        reasoning=final_reasoning,
        confidence_score=final_confidence,
        needs_human_review=needs_review,
        anomaly_details=anomaly_details,
        alert_sent=alert_sent,
        alert_details=alert_details
    )

@app.post("/expenses/{expense_id}/review")
async def review_expense(expense_id: int, decision: ApprovalDecision):
    """Human review endpoint for expenses with optional notifications"""
    
    if expense_id > len(expenses_df) or expense_id < 1:
        raise HTTPException(status_code=404, detail="Expense not found")
    
    expense = expenses_df.iloc[expense_id - 1]
    
    if expense['status'] not in ['pending', 'escalated', 'suspicious', 'budget_exceeded']:
        raise HTTPException(status_code=400, detail="Expense does not require review or already processed")
    
    # Update expense status
    new_status = "approved" if decision.approved else "rejected"
    expenses_df.at[expense_id - 1, 'status'] = new_status
    
    # Update budget if approved
    if decision.approved:
        dept = expense['department']
        amount = expense['amount']
        departments_df.loc[departments_df['department'] == dept, 'budget_usage'] += amount
    
    # Send notification if requested
    if decision.send_notification:
        # Get employee manager phone (in production, you'd have an employee-manager mapping)
        dept_data = departments_df[departments_df['department'] == expense['department']]
        if not dept_data.empty and 'manager_phone' in dept_data.columns:
            manager_phone = dept_data.iloc[0]['manager_phone']
            
            expense_data = {
                'expense_id': expense_id,
                'amount': expense['amount'],
                'reviewer': decision.reviewer,
                'comments': decision.comments
            }
            
            alert_type = AlertType.EXPENSE_APPROVED if decision.approved else AlertType.EXPENSE_REJECTED
            
            # Send alert in background
            asyncio.create_task(
                messaging_service.send_alert(
                    alert_type=alert_type,
                    recipient_phone=manager_phone,
                    expense_data=expense_data
                )
            )
    
    # Record decision
    reasoning = f"Manual review: {'approved' if decision.approved else 'rejected'} by {decision.reviewer}"
    if decision.comments:
        reasoning += f". Comments: {decision.comments}"
    
    LearningGovernanceService.record_decision(
        expense_id, f"review_{new_status}", reasoning, decision.reviewer, 
        decision.comments, {"notification_sent": decision.send_notification}
    )
    
    return {
        "message": f"Expense {expense_id} {new_status}",
        "expense_id": expense_id,
        "reviewer": decision.reviewer,
        "status": new_status,
        "comments": decision.comments,
        "notification_sent": decision.send_notification
    }

@app.get("/cfo/report")
async def get_cfo_report(background_tasks: BackgroundTasks):
    """Generate CFO summary report with alert notification"""
    report = CFOReportService.generate_report()
    
    # Notify CFO if high risk in background
    if report['risk_assessment']['risk_level'] in ['HIGH', 'MEDIUM']:
        background_tasks.add_task(NotificationService.notify_cfo, report)
    
    return report

@app.get("/departments/")
async def get_departments():
    """Get department information and budgets"""
    return departments_df.to_dict('records')

@app.put("/departments/{department}")
async def update_department(department: str, update: DepartmentUpdate):
    """Update department policies and limits"""
    
    if department not in departments_df['department'].values:
        raise HTTPException(status_code=404, detail="Department not found")
    
    # Update specified fields
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            departments_df.loc[departments_df['department'] == department, field] = value
    
    # Record policy update
    LearningGovernanceService.record_decision(
        None, "policy_update", f"Updated policies for {department}", "admin", 
        f"Updated fields: {list(update_data.keys())}"
    )
    
    return {
        "message": f"Department {department} updated successfully",
        "updated_fields": update_data,
        "current_state": departments_df[departments_df['department'] == department].iloc[0].to_dict()
    }

@app.get("/expenses/")
async def get_expenses(
    status: Optional[str] = Query(None, description="Filter by status"),
    department: Optional[str] = Query(None, description="Filter by department"),
    employee_id: Optional[str] = Query(None, regex=r'^E\d{3}$', description="Filter by employee ID"),
    min_amount: Optional[float] = Query(None, ge=0, description="Minimum amount"),
    max_amount: Optional[float] = Query(None, ge=0, description="Maximum amount"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size")
):
    """Get expenses with filtering and pagination"""
    filtered_df = expenses_df.copy()
    
    if status:
        filtered_df = filtered_df[filtered_df['status'] == status]
    if department:
        filtered_df = filtered_df[filtered_df['department'] == department]
    if employee_id:
        filtered_df = filtered_df[filtered_df['employee_id'] == employee_id]
    if min_amount is not None:
        filtered_df = filtered_df[filtered_df['amount'] >= min_amount]
    if max_amount is not None:
        filtered_df = filtered_df[filtered_df['amount'] <= max_amount]
    
    # Pagination
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_df = filtered_df.iloc[start_idx:end_idx]
    
    return {
        "data": paginated_df.to_dict('records'),
        "pagination": {
            "page": page,
            "size": size,
            "total": len(filtered_df),
            "pages": (len(filtered_df) + size - 1) // size
        }
    }

@app.get("/audit/trail")
async def get_audit_trail(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=100)
):
    """Get audit trail for governance and transparency"""
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    
    paginated_trail = audit_trail[start_idx:end_idx]
    
    # Create a copy for serialization without modifying the original
    serializable_trail = []
    for record in paginated_trail:
        serializable_record = record.copy()
        # Convert datetime to string for JSON serialization
        serializable_record['timestamp'] = serializable_record['timestamp'].isoformat()
        serializable_trail.append(serializable_record)
    
    return {
        "data": serializable_trail,
        "pagination": {
            "page": page,
            "size": size,
            "total": len(audit_trail),
            "pages": (len(audit_trail) + size - 1) // size
        },
        "statistics": LearningGovernanceService.get_audit_stats()
    }

# Messaging Endpoints
@app.post("/alerts/send")
async def send_custom_alert(alert_request: AlertRequest):
    """Send a custom alert to a specific recipient"""
    
    # Validate department if provided
    if alert_request.department and alert_request.department not in departments_df['department'].values:
        raise HTTPException(status_code=400, detail="Invalid department")
    
    # Get expense data if expense_id is provided
    expense_data = None
    if alert_request.expense_id:
        if alert_request.expense_id > len(expenses_df) or alert_request.expense_id < 1:
            raise HTTPException(status_code=404, detail="Expense not found")
        expense_data = expenses_df.iloc[alert_request.expense_id - 1].to_dict()
    
    # Send alert
    result = await messaging_service.send_alert(
        alert_type=alert_request.alert_type,
        recipient_phone=alert_request.recipient_phone,
        expense_data=expense_data,
        department_data=departments_df[departments_df['department'] == alert_request.department].iloc[0].to_dict() if alert_request.department else None
    )
    
    return {
        "message": "Alert sent successfully",
        "result": result,
        "alert_type": alert_request.alert_type.value,
        "recipient": alert_request.recipient_phone
    }

@app.get("/messaging/status")
async def get_messaging_status():
    """Get the status of the messaging service"""
    return {
        "service": "Twilio Messaging",
        "configured": messaging_service.is_configured,
        "status": "operational" if messaging_service.is_configured else "not_configured",
        "capabilities": {
            "whatsapp": bool(messaging_service.whatsapp_from),
            "sms": bool(messaging_service.sms_from),
            "templates": bool(messaging_service.content_sid)
        }
    }

@app.get("/alerts/history")
async def get_alert_history(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100)
):
    """Get history of sent alerts"""
    # Filter audit trail for alert entries
    alert_records = [record for record in audit_trail if 'alert_type' in record]
    
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_alerts = alert_records[start_idx:end_idx]
    
    # Create a copy for serialization
    serializable_alerts = []
    for record in paginated_alerts:
        serializable_record = record.copy()
        serializable_record['timestamp'] = serializable_record['timestamp'].isoformat()
        serializable_alerts.append(serializable_record)
    
    return {
        "data": serializable_alerts,
        "pagination": {
            "page": page,
            "size": size,
            "total": len(alert_records),
            "pages": (len(alert_records) + size - 1) // size
        },
        "summary": {
            "total_alerts": len(alert_records),
            "alerts_by_type": pd.DataFrame(alert_records)['alert_type'].value_counts().to_dict() if alert_records else {}
        }
    }

# ML Endpoints
@app.post("/ml/retrain", response_model=MLTrainingResponse)
async def retrain_ml_model():
    """Retrain the ML anomaly detection model"""
    try:
        result = ml_detector.train_ml_model(expenses_df)
        
        if result["status"] == "success":
            logger.info(f"ML model retrained successfully on {result['training_samples']} samples")
            return MLTrainingResponse(
                status="success",
                training_samples=result["training_samples"],
                anomalies_detected=result["anomalies_detected"]
            )
        else:
            return MLTrainingResponse(status=result["status"], error=result.get("error"))
            
    except Exception as e:
        logger.error(f"ML model retraining failed: {e}")
        return MLTrainingResponse(status="error", error=str(e))

@app.get("/ml/info")
async def get_ml_info():
    """Get information about the ML model"""
    detector_info = ml_detector.get_detector_info()
    
    # Add performance metrics
    if len(expenses_df) > 0:
        anomaly_stats = {
            "total_expenses": len(expenses_df),
            "anomalies_detected": int(expenses_df['ml_anomaly'].sum()),
            "anomaly_rate": float((expenses_df['ml_anomaly'].sum() / len(expenses_df)) * 100),
            "avg_confidence": float(expenses_df['anomaly_confidence'].mean())
        }
        detector_info["performance"] = anomaly_stats
    
    return detector_info

@app.get("/ml/features")
async def get_ml_features():
    """Get information about ML features"""
    if not ml_detector.ml_detector.is_trained:
        return {"message": "ML model not trained yet"}
    
    return {
        "feature_count": len(ml_detector.ml_detector.feature_names),
        "features": ml_detector.ml_detector.feature_names,
        "feature_categories": {
            "basic_amount": ["amount", "log_amount", "sqrt_amount"],
            "employee_behavior": ["emp_submission_count", "emp_mean_amount", "emp_std_amount", 
                                "emp_median_amount", "emp_max_amount", "emp_z_score_mean", "emp_z_score_median"],
            "department_behavior": ["dept_submission_count", "dept_mean_amount", "dept_std_amount",
                                  "dept_median_amount", "dept_z_score", "dept_95th_percentile"],
            "purpose_temporal": ["purpose_encoded", "day_of_month", "day_of_week"],
            "interaction": ["amount_purpose_interaction", "amount_dept_ratio"]
        }
    }

# Background tasks
async def retrain_ml_model_background():
    """Background task to retrain ML model"""
    try:
        result = ml_detector.train_ml_model(expenses_df)
        if result["status"] == "success":
            logger.info(f"Background ML retraining completed: {result}")
        else:
            logger.warning(f"Background ML retraining issues: {result}")
    except Exception as e:
        logger.error(f"Background ML retraining failed: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system and train ML model on startup"""
    logger.info("Expense Management API with ML Anomaly Detection & Alert System starting up...")
    logger.info(f"Loaded {len(expenses_df)} existing expenses")
    logger.info(f"Tracking {len(departments_df)} departments")
    logger.info(f"Messaging service: {'CONFIGURED' if messaging_service.is_configured else 'SIMULATION MODE'}")
    
    # Train ML model on historical data
    try:
        result = ml_detector.train_ml_model(expenses_df)
        if result["status"] == "success":
            logger.info(f"ML model trained successfully on {result['training_samples']} samples")
        else:
            logger.warning(f"ML model training completed with status: {result}")
    except Exception as e:
        logger.warning(f"Initial ML model training failed: {e}. Using rule-based fallback.")
    
    logger.info("System startup completed")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": {
                "expenses": len(expenses_df),
                "departments": len(departments_df),
                "audit_entries": len(audit_trail)
            },
            "ml_model": {
                "trained": ml_detector.ml_detector.is_trained,
                "status": "operational"
            },
            "messaging": {
                "configured": messaging_service.is_configured,
                "status": "operational" if messaging_service.is_configured else "simulation"
            },
            "system": {
                "version": "2.1.0",
                "uptime": "running"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True  # Enable auto-reload for development
    )
