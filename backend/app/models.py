from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from datetime import datetime
from app.database import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    quality_score = Column(Integer)
    quality_label = Column(String)
    
    # Store issues as a JSON structure
    issues = Column(JSON)
    # Store stats/features as a JSON structure
    stats = Column(JSON)
