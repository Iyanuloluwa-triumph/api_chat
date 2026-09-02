from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from groq import Groq

load_dotenv(find_dotenv())

# APIRouter definition
router = APIRouter(prefix="/organization", tags=["Organization & Seller Admin"])

# Initialize Groq Client securely from environment variables
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Pydantic Schemas
class ProductCreate(BaseModel):
    name: str
    description: str
    price: float
    category: str
    stock_quantity: int

class ProductResponse(ProductCreate):
    id: str
    created_at: str

class AnalyticsPayload(BaseModel):
    organization_id: Optional[str] = None
    seller_id: Optional[str] = None
    metrics: Dict[str, Any]

# ----------------------------------------------------
# 1. Product Upload Route
# ----------------------------------------------------
@router.post("/products", response_model=ProductResponse)
async def upload_product(
    name: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
    stock_quantity: int = Form(...),
    image: Optional[UploadFile] = File(None)
):
    try:
        product_id = f"prod_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Read image contents if provided (e.g., for cloud storage upload or ML preprocessing)
        if image:
            image_bytes = await image.read()
            await image.close()

        product_data = {
            "id": product_id,
            "name": name,
            "description": description,
            "price": price,
            "category": category,
            "stock_quantity": stock_quantity,
            "created_at": datetime.now().isoformat()
        }
        
        return product_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create product: {str(e)}")

# ----------------------------------------------------
# 2. Dynamic Sales Summary & AI Report Generation Route
# ----------------------------------------------------
@router.post("/reports/weekly")
async def generate_weekly_summary_report(payload: AnalyticsPayload):
    """
    Receives dynamic JSON aggregated from PostgreSQL via the Java backend 
    and generates an AI summary report using Groq.
    """
    metrics_dict = payload.metrics
    
    if not metrics_dict:
        raise HTTPException(status_code=400, detail="Empty sales metrics dictionary received.")

    prompt = f"""
    You are an expert e-commerce business analyst. Write a concise, professional weekly performance report for a seller based on these PostgreSQL sales metrics:
    
    {json.dumps(metrics_dict, indent=2)}
    
    Format the response in clean Markdown with these sections:
    -  Executive Overview
    -  Top Performers & Wins
    -  Actionable Inventory Alerts
    -  Strategic Recommendations for Next Week
    """
    
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        ai_narrative = completion.choices[0].message.content
        
        return {
            "organization_id": payload.organization_id,
            "seller_id": payload.seller_id,
            "metrics": metrics_dict,
            "ai_report": ai_narrative
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")