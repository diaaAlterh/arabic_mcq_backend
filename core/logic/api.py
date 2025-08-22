from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import json
import tempfile
import shutil
from datetime import datetime
import uuid
import asyncio
from pathlib import Path

# استيراد الكلاسات الأساسية من الكود الأصلي
from main import ArabicMCQGeneratorSystem

app = FastAPI(
    title="Arabic MCQ Generator API",
    description="API لتوليد أسئلة الاختيار من متعدد باللغة العربية",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج، حدد النطاقات المسموحة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# مجلدات التخزين المؤقت
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# تهيئة النظام
generator = ArabicMCQGeneratorSystem()

# نماذج البيانات
class TextInput(BaseModel):
    text: str = Field(..., min_length=50, description="النص المراد توليد أسئلة منه")
    question_count: int = Field(5, ge=1, le=50, description="عدد الأسئلة المطلوبة")

class MCQQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: Optional[str] = None
    difficulty: Optional[str] = None

class MCQResponse(BaseModel):
    questions: List[MCQQuestion]
    metadata: Dict[str, Any]

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# تخزين حالة المهام
tasks_storage: Dict[str, TaskStatus] = {}

def cleanup_old_files():
    """تنظيف الملفات القديمة (أكثر من ساعة)"""
    try:
        current_time = datetime.now()
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                file_age = current_time.timestamp() - file_path.stat().st_mtime
                if file_age > 3600:  # ساعة واحدة
                    file_path.unlink()
        
        for file_path in OUTPUT_DIR.iterdir():
            if file_path.is_file():
                file_age = current_time.timestamp() - file_path.stat().st_mtime
                if file_age > 3600:
                    file_path.unlink()
    except Exception as e:
        print(f"خطأ في تنظيف الملفات: {e}")

@app.on_event("startup")
async def startup_event():
    """تشغيل المهام عند بدء التطبيق"""
    print("🚀 بدء تشغيل Arabic MCQ Generator API")
    
    # التحقق من API Key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ تحذير: GEMINI_API_KEY غير موجود")
    
    # تنظيف الملفات القديمة
    cleanup_old_files()

@app.get("/", summary="الصفحة الرئيسية")
async def root():
    return {
        "message": "مرحباً بك في API توليد أسئلة الاختيار من متعدد",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", summary="فحص صحة الخدمة")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY"))
    }

@app.post("/generate/text", response_model=Dict[str, str], summary="توليد أسئلة من النص")
async def generate_from_text(
    background_tasks: BackgroundTasks,
    input_data: TextInput
):
    """توليد أسئلة الاختيار من متعدد من النص المدخل"""
    
    task_id = str(uuid.uuid4())
    task_status = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=datetime.now()
    )
    tasks_storage[task_id] = task_status
    
    # بدء المعالجة في الخلفية
    background_tasks.add_task(
        process_text_generation,
        task_id,
        input_data.text,
        input_data.question_count
    )
    
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "تم قبول المهمة وبدء المعالجة"
    }

@app.post("/generate/image", response_model=Dict[str, str], summary="توليد أسئلة من الصورة")
async def generate_from_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ملف الصورة"),
    question_count: int = Form(5, ge=1, le=50, description="عدد الأسئلة")
):
    """توليد أسئلة من صورة تحتوي على نص"""
    
    # التحقق من نوع الملف
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="نوع الملف غير مدعوم. الأنواع المدعومة: JPEG, PNG, GIF, BMP, TIFF, WEBP"
        )
    
    # حفظ الملف مؤقتاً
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في حفظ الملف: {str(e)}")
    
    task_id = str(uuid.uuid4())
    task_status = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=datetime.now()
    )
    tasks_storage[task_id] = task_status
    
    # بدء المعالجة في الخلفية
    background_tasks.add_task(
        process_image_generation,
        task_id,
        str(file_path),
        question_count
    )
    
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "تم رفع الملف وبدء المعالجة"
    }

@app.post("/generate/pdf", response_model=Dict[str, str], summary="توليد أسئلة من PDF")
async def generate_from_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ملف PDF"),
    question_count: int = Form(5, ge=1, le=50, description="عدد الأسئلة"),
    page_range: str = Form("all", description="نطاق الصفحات (مثال: all أو 1-5 أو 1,3,5)")
):
    """توليد أسئلة من ملف PDF"""
    
    # التحقق من نوع الملف
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف بصيغة PDF")
    
    # حفظ الملف مؤقتاً
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في حفظ الملف: {str(e)}")
    
    task_id = str(uuid.uuid4())
    task_status = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=datetime.now()
    )
    tasks_storage[task_id] = task_status
    
    # بدء المعالجة في الخلفية
    background_tasks.add_task(
        process_pdf_generation,
        task_id,
        str(file_path),
        question_count,
        page_range
    )
    
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "تم رفع ملف PDF وبدء المعالجة"
    }

@app.get("/task/{task_id}", response_model=TaskStatus, summary="فحص حالة المهمة")
async def get_task_status(task_id: str):
    """الحصول على حالة المهمة"""
    
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    
    return tasks_storage[task_id]

@app.get("/download/{task_id}/word", summary="تحميل ملف Word")
async def download_word_file(task_id: str):
    """تحميل ملف Word للمهمة المكتملة"""
    
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    
    task = tasks_storage[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="المهمة لم تكتمل بعد")
    
    if not task.result or "word_file" not in task.result:
        raise HTTPException(status_code=404, detail="ملف Word غير متوفر")
    
    file_path = Path(task.result["word_file"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

@app.get("/download/{task_id}/json", summary="تحميل ملف JSON")
async def download_json_file(task_id: str):
    """تحميل ملف JSON للمهمة المكتملة"""
    
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    
    task = tasks_storage[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="المهمة لم تكتمل بعد")
    
    if not task.result:
        raise HTTPException(status_code=404, detail="النتائج غير متوفرة")
    
    return JSONResponse(content=task.result["questions_data"])

@app.get("/pdf/info", summary="معلومات ملف PDF")
async def get_pdf_info(file: UploadFile = File(...)):
    """الحصول على معلومات ملف PDF قبل المعالجة"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف بصيغة PDF")
    
    # حفظ الملف مؤقتاً
    file_path = UPLOAD_DIR / f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # الحصول على معلومات PDF
        pdf_info = generator.get_pdf_info(str(file_path))
        
        # حذف الملف المؤقت
        file_path.unlink()
        
        return pdf_info
        
    except Exception as e:
        # حذف الملف في حالة الخطأ
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة PDF: {str(e)}")

@app.delete("/task/{task_id}", summary="حذف المهمة")
async def delete_task(task_id: str):
    """حذف المهمة وملفاتها المرتبطة"""
    
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    
    task = tasks_storage[task_id]
    
    # حذف الملفات المرتبطة
    if task.result and "word_file" in task.result:
        word_file = Path(task.result["word_file"])
        if word_file.exists():
            word_file.unlink()
    
    # حذف المهمة من التخزين
    del tasks_storage[task_id]
    
    return {"message": "تم حذف المهمة بنجاح"}

@app.get("/tasks", summary="قائمة جميع المهام")
async def list_tasks():
    """الحصول على قائمة بجميع المهام"""
    
    tasks_list = []
    for task_id, task in tasks_storage.items():
        tasks_list.append({
            "task_id": task_id,
            "status": task.status,
            "progress": task.progress,
            "created_at": task.created_at,
            "completed_at": task.completed_at
        })
    
    return {"tasks": tasks_list, "total": len(tasks_list)}

# وظائف المعالجة في الخلفية
async def process_text_generation(task_id: str, text: str, question_count: int):
    """معالجة توليد الأسئلة من النص"""
    
    try:
        # تحديث حالة المهمة
        tasks_storage[task_id].status = "processing"
        tasks_storage[task_id].progress = 10
        
        # تشغيل توليد الأسئلة
        result = generator.run_mcq_generation(text, 'text', question_count)
        
        tasks_storage[task_id].progress = 70
        
        if "error" in result:
            tasks_storage[task_id].status = "failed"
            tasks_storage[task_id].error = result["error"]
            return
        
        # حفظ النتائج
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"mcq_text_{task_id}_{timestamp}.docx"
        
        success = generator.save_mcq_to_word(result, str(filename))
        
        if not success:
            tasks_storage[task_id].status = "failed"
            tasks_storage[task_id].error = "فشل في حفظ الملف"
            return
        
        tasks_storage[task_id].progress = 100
        tasks_storage[task_id].status = "completed"
        tasks_storage[task_id].completed_at = datetime.now()
        tasks_storage[task_id].result = {
            "questions_data": result,
            "word_file": str(filename),
            "question_count": len(result.get("questions", []))
        }
        
    except Exception as e:
        tasks_storage[task_id].status = "failed"
        tasks_storage[task_id].error = str(e)

async def process_image_generation(task_id: str, image_path: str, question_count: int):
    """معالجة توليد الأسئلة من الصورة"""
    
    try:
        tasks_storage[task_id].status = "processing"
        tasks_storage[task_id].progress = 10
        
        result = generator.run_mcq_generation(image_path, 'image', question_count)
        
        tasks_storage[task_id].progress = 70
        
        if "error" in result:
            tasks_storage[task_id].status = "failed"
            tasks_storage[task_id].error = result["error"]
            # حذف الملف المؤقت
            Path(image_path).unlink(missing_ok=True)
            return
        
        # حفظ النتائج
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"mcq_image_{task_id}_{timestamp}.docx"
        
        success = generator.save_mcq_to_word(result, str(filename))
        
        # حذف الملف المؤقت
        Path(image_path).unlink(missing_ok=True)
        
        if not success:
            tasks_storage[task_id].status = "failed"
            tasks_storage[task_id].error = "فشل في حفظ الملف"
            return
        
        tasks_storage[task_id].progress = 100
        tasks_storage[task_id].status = "completed"
        tasks_storage[task_id].completed_at = datetime.now()
        tasks_storage[task_id].result = {
            "questions_data": result,
            "word_file": str(filename),
            "question_count": len(result.get("questions", []))
        }
        
    except Exception as e:
        tasks_storage[task_id].status = "failed"
        tasks_storage[task_id].error = str(e)
        # حذف الملف المؤقت في حالة الخطأ
        Path(image_path).unlink(missing_ok=True)

async def process_pdf_generation(task_id: str, pdf_path: str, question_count: int, page_range: str):
    """معالجة توليد الأسئلة من PDF"""
    
    try:
        tasks_storage[task_id].status = "processing"
        tasks_storage[task_id].progress = 10
        
        result = generator.run_mcq_generation(pdf_path, 'pdf', question_count, page_range=page_range)
        
        tasks_storage[task_id].progress = 70
        
        if "error" in result:
            tasks_storage[task_id].status = "failed"
            tasks_storage[task_id].error = result["error"]
            # حذف الملف المؤقت
            Path(pdf_path).unlink(missing_ok=True)
            return
        
        # حفظ النتائج
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUT_DIR / f"mcq_pdf_{task_id}_{timestamp}.docx"
        
        success = generator.save_mcq_to_word(result, str(filename))
        
        # حذف الملف المؤقت
        Path(pdf_path).unlink(missing_ok=True)
        
        if not success:
            tasks_storage[task_id].status = "failed"
            tasks_storage[task_id].error = "فشل في حفظ الملف"
            return
        
        tasks_storage[task_id].progress = 100
        tasks_storage[task_id].status = "completed"
        tasks_storage[task_id].completed_at = datetime.now()
        tasks_storage[task_id].result = {
            "questions_data": result,
            "word_file": str(filename),
            "question_count": len(result.get("questions", []))
        }
        
    except Exception as e:
        tasks_storage[task_id].status = "failed"
        tasks_storage[task_id].error = str(e)
        # حذف الملف المؤقت في حالة الخطأ
        Path(pdf_path).unlink(missing_ok=True)

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))  # خذ رقم المنفذ من المتغير البيئي أو 8000 كافتراضي

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
