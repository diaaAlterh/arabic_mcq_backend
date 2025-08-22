from django.urls import path
from .views import GenerateMCQs, UserMCQRequestsView
from .views import (
    GenerateFromTextView, GenerateFromImageView, GenerateFromPDFView,
    TaskStatusView, DownloadWordView, DownloadJSONView, DeleteTaskView,
    ListTasksView, PDFInfoView
)

urlpatterns = [
    path('generate-mcqs/', GenerateMCQs.as_view(), name='generate-mcqs'),
    path('my-mcq-requests/', UserMCQRequestsView.as_view(), name='my-mcq-requests'), # New URL for retrieving requests
    path("generate/text/", GenerateFromTextView.as_view(), name="generate_text"),
    path("generate/image/", GenerateFromImageView.as_view(), name="generate_image"),
    path("generate/pdf/", GenerateFromPDFView.as_view(), name="generate_pdf"),
    path("pdf/info/", PDFInfoView.as_view(), name="pdf_info"),

    path("task/<uuid:task_id>/", TaskStatusView.as_view(), name="task_status"),
    path("download/<uuid:task_id>/word/", DownloadWordView.as_view(), name="download_word"),
    path("download/<uuid:task_id>/json/", DownloadJSONView.as_view(), name="download_json"),
    path("task/<uuid:task_id>/delete/", DeleteTaskView.as_view(), name="delete_task"),

    path("tasks/", ListTasksView.as_view(), name="list_tasks"),

]