import logging # Import for logging errors

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

from core.logic.crew_engine import run_mcq_pipeline
from .models import MCQRequest # Import the new model
from .serializers import MCQRequestSerializer # Import the new serializer
import threading
from datetime import datetime
from pathlib import Path
from django.http import FileResponse, JsonResponse, Http404

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Task
from .serializers import TextInputSerializer, TaskSerializer

# Import your MCQ generator
from core.logic.main import ArabicMCQGeneratorSystem
import uuid
import mimetypes



logger = logging.getLogger(__name__) # Initialize logger
generator = ArabicMCQGeneratorSystem()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class GenerateMCQs(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file = request.FILES.get("file")
        text = request.data.get("text")

        if not file and not text:
            return Response(
                {"error": "You must provide either an image, a PDF, or text"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if file and text:
            return Response(
                {"error": "Provide only one of: image, PDF, or text"},
                status=status.HTTP_400_BAD_REQUEST
            )

        final_text = None

        if file:
            file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.name}"
            with open(file_path, "wb+") as dest:
                for chunk in file.chunks():
                    dest.write(chunk)

            # Detect MIME type and fallback to extension check
            mime_type, _ = mimetypes.guess_type(file_path)
            ext = file_path.suffix.lower()

            allowed_image_exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
            pdf_ext = ".pdf"

            if (mime_type and mime_type.startswith("image")) or ext in allowed_image_exts:
                print("🔍 Extracting text from image...")
                extracted_text = generator.extract_text_from_image(str(file_path))
                if extracted_text.startswith("خطأ"):
                    return Response({'error': extracted_text}, status=400)
                print(f"✅ Text extracted successfully ({len(extracted_text)} chars)")
                final_text = extracted_text

            elif (mime_type == "application/pdf") or ext == pdf_ext:
                print("📄 Extracting text from PDF...")
                extracted_text = generator.extract_text_from_pdf(str(file_path))
                if extracted_text.startswith("خطأ"):
                    return Response({'error': extracted_text}, status=400)
                print(f"✅ Text extracted successfully ({len(extracted_text)} chars)")
                final_text = extracted_text

            else:
                return Response({"detail": "Unsupported file type"}, status=400)

        if text:
            final_text = text

        if final_text:
            try:
                # 1. Run the MCQ generation pipeline
                result = run_mcq_pipeline(final_text)

                # 2. Save the request and its result to the database
                mcq_request = MCQRequest.objects.create(
                    user=request.user,
                    input_text=final_text,
                    generated_mcqs=result
                )

                # 3. Return the generated MCQs to the user
                return Response(result, status=status.HTTP_200_OK)

            except Exception:
                logger.exception("Error occurred while generating MCQs")
                return Response({"error": "An internal server error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            
class Extract(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file = request.FILES.get("file")
        text = request.data.get("text")

        if not file and not text:
            return Response(
                {"error": "You must provide either an image, a PDF, or text"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # ممنوع يجي أكتر من وحدة بنفس الوقت
        if file and text:
            return Response(
                {"error": "Provide only one of: image, PDF, or text"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # معالجة الملف
        if file:
            allowed_image_types = [
                "image/jpeg", "image/png", "image/gif",
                "image/bmp", "image/tiff", "image/webp"
            ]
            pdf_type = "application/pdf"
            file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.name}"
            with open(file_path, "wb+") as dest:
                for chunk in file.chunks():
                    dest.write(chunk)

            if file.content_type in allowed_image_types:
                        
                print("🔍 جاري استخراج النص من الصورة...")
                confidence = generator.get_text_confidence(str(file_path))
                print(f"مستوى ثقة OCR: {confidence['average_confidence']:.1f}%")
                
                extracted_text = generator.extract_text_from_image(str(file_path))
                if extracted_text.startswith("خطأ"):
                    return Response({'error': extracted_text, 'confidence': confidence},status=400)

                    
                print(f"تم استخراج النص بنجاح ({len(extracted_text)} حرف)")
                final_text = extracted_text
                

            elif file.content_type == pdf_type:
                        
                print("📄 جاري استخراج النص من PDF...")
                extracted_text = generator.extract_text_from_pdf(str(file_path))
                
                if extracted_text.startswith("خطأ"):
                    return Response({'error': extracted_text},status=400)
                    
                print(f"تم استخراج النص بنجاح ({len(extracted_text)} حرف)")
                final_text = extracted_text


            else:
                return Response({"detail": "Unsupported file type"}, status=400)

        # معالجة النص
        if text:
            final_text=text
            
        if final_text:
            try:
                return Response({"extracted text":final_text}, status=status.HTTP_200_OK)

            except Exception as e:
                # Log the full traceback for debugging purposes
                logger.exception("Error occurred while Extracting Text")  # This logs full traceback
                return Response({"error": "An internal server error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserMCQRequestsView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Retrieve all MCQ requests made by the current authenticated user
        # They are automatically ordered by 'created_at' due to Meta.ordering in the model
        mcq_requests = MCQRequest.objects.filter(user=request.user)
        
        # Serialize the queryset of MCQRequest objects
        serializer = MCQRequestSerializer(mcq_requests, many=True) # many=True because we're serializing a list
        
        return Response(serializer.data, status=status.HTTP_200_OK)
    

# ---------------- Generate from text ----------------

class GenerateFromTextView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        serializer = TextInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        task = Task.objects.create(status="pending")

        threading.Thread(
            target=self.process_text_generation,
            args=(task.id, data["text"], data["question_count"])
        ).start()

        return Response({
            "task_id": str(task.id),
            "status": "accepted",
            "message": "Task accepted and processing started"
        }, status=status.HTTP_202_ACCEPTED)

    def process_text_generation(self, task_id, text, question_count):
        task = Task.objects.get(id=task_id)
        try:
            task.status = "processing"
            task.progress = 10
            task.save()

            result = generator.run_mcq_generation(text, 'text', question_count)
            task.progress = 70

            if "error" in result:
                task.status = "failed"
                task.error = result["error"]
                task.save()
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = OUTPUT_DIR / f"mcq_text_{task_id}_{timestamp}.docx"

            success = generator.save_mcq_to_word(result, str(filename))
            if not success:
                task.status = "failed"
                task.error = "Failed to save Word file"
                task.save()
                return

            task.status = "completed"
            task.progress = 100
            task.completed_at = datetime.now()
            task.result = {
                "questions_data": result,
                "word_file": str(filename),
                "question_count": len(result.get("questions", []))
            }
            task.save()
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.save()

# ---------------- Generate from image ----------------

class GenerateFromImageView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        file = request.FILES.get("file")
        question_count = int(request.POST.get("question_count", 5))

        if not file:
            return Response({"detail": "No file uploaded"}, status=400)

        allowed_types = [
            "image/jpeg", "image/png", "image/gif",
            "image/bmp", "image/tiff", "image/webp"
        ]
        if file.content_type not in allowed_types:
            return Response({"detail": "Unsupported file type"}, status=400)

        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.name}"
        with open(file_path, "wb+") as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        task = Task.objects.create(status="pending")

        threading.Thread(
            target=self.process_image_generation,
            args=(task.id, str(file_path), question_count)
        ).start()

        return Response({
            "task_id": str(task.id),
            "status": "accepted",
            "message": "File uploaded and processing started"
        }, status=status.HTTP_202_ACCEPTED)

    def process_image_generation(self, task_id, image_path, question_count):
        task = Task.objects.get(id=task_id)
        try:
            task.status = "processing"
            task.progress = 10
            task.save()

            result = generator.run_mcq_generation(image_path, 'image', question_count)
            task.progress = 70

            if "error" in result:
                task.status = "failed"
                task.error = result["error"]
                task.save()
                Path(image_path).unlink(missing_ok=True)
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = OUTPUT_DIR / f"mcq_image_{task_id}_{timestamp}.docx"

            success = generator.save_mcq_to_word(result, str(filename))
            Path(image_path).unlink(missing_ok=True)

            if not success:
                task.status = "failed"
                task.error = "Failed to save Word file"
                task.save()
                return

            task.status = "completed"
            task.progress = 100
            task.completed_at = datetime.now()
            task.result = {
                "questions_data": result,
                "word_file": str(filename),
                "question_count": len(result.get("questions", []))
            }
            task.save()
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.save()
            Path(image_path).unlink(missing_ok=True)

# ---------------- Generate from PDF ----------------

class GenerateFromPDFView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        file = request.FILES.get("file")
        question_count = int(request.POST.get("question_count", 5))
        page_range = request.POST.get("page_range", "all")

        if not file:
            return Response({"detail": "No file uploaded"}, status=400)

        if not file.name.lower().endswith(".pdf"):
            return Response({"detail": "File must be PDF"}, status=400)

        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.name}"
        with open(file_path, "wb+") as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        task = Task.objects.create(status="pending")

        threading.Thread(
            target=self.process_pdf_generation,
            args=(task.id, str(file_path), question_count, page_range)
        ).start()

        return Response({
            "task_id": str(task.id),
            "status": "accepted",
            "message": "PDF uploaded and processing started"
        }, status=status.HTTP_202_ACCEPTED)

    def process_pdf_generation(self, task_id, pdf_path, question_count, page_range):
        task = Task.objects.get(id=task_id)
        try:
            task.status = "processing"
            task.progress = 10
            task.save()

            result = generator.run_mcq_generation(pdf_path, 'pdf', question_count, page_range=page_range)
            task.progress = 70

            if "error" in result:
                task.status = "failed"
                task.error = result["error"]
                task.save()
                Path(pdf_path).unlink(missing_ok=True)
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = OUTPUT_DIR / f"mcq_pdf_{task_id}_{timestamp}.docx"

            success = generator.save_mcq_to_word(result, str(filename))
            Path(pdf_path).unlink(missing_ok=True)

            if not success:
                task.status = "failed"
                task.error = "Failed to save Word file"
                task.save()
                return

            task.status = "completed"
            task.progress = 100
            task.completed_at = datetime.now()
            task.result = {
                "questions_data": result,
                "word_file": str(filename),
                "question_count": len(result.get("questions", []))
            }
            task.save()
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.save()
            Path(pdf_path).unlink(missing_ok=True)

# ---------------- PDF info ----------------

class PDFInfoView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        file = request.FILES.get("file")
        if not file:
            return Response({"detail": "No file uploaded"}, status=400)

        if not file.name.lower().endswith(".pdf"):
            return Response({"detail": "File must be PDF"}, status=400)

        file_path = UPLOAD_DIR / f"temp_{uuid.uuid4()}_{file.name}"
        try:
            with open(file_path, "wb+") as dest:
                for chunk in file.chunks():
                    dest.write(chunk)

            pdf_info = generator.get_pdf_info(str(file_path))
            file_path.unlink(missing_ok=True)
            return Response(pdf_info)
        except Exception as e:
            file_path.unlink(missing_ok=True)
            return Response({"detail": str(e)}, status=500)

# ---------------- Task status & management ----------------

class TaskStatusView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request, task_id):
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            raise Http404("Task not found")
        return Response(TaskSerializer(task).data)

class DownloadWordView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request, task_id):
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            raise Http404("Task not found")

        if task.status != "completed" or not task.result or "word_file" not in task.result:
            return Response({"detail": "Word file not available"}, status=400)

        file_path = Path(task.result["word_file"])
        if not file_path.exists():
            raise Http404("File not found")

        return FileResponse(open(file_path, "rb"),
            as_attachment=True,
            filename=file_path.name,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

class DownloadJSONView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request, task_id):
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            raise Http404("Task not found")

        if task.status != "completed" or not task.result:
            return Response({"detail": "JSON not available"}, status=400)

        return JsonResponse(task.result["questions_data"], safe=False)

class DeleteTaskView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def delete(self, request, task_id):
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            raise Http404("Task not found")

        if task.result and "word_file" in task.result:
            file_path = Path(task.result["word_file"])
            file_path.unlink(missing_ok=True)

        task.delete()
        return Response({"message": "Task deleted successfully"})

class ListTasksView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        tasks = Task.objects.all().order_by("-created_at")
        return Response({
            "tasks": TaskSerializer(tasks, many=True).data,
            "total": tasks.count()
        })
