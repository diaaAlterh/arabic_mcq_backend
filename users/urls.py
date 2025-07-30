from django.urls import path
from core.views import GenerateMCQs, UserMCQRequestsView

urlpatterns = [
    path('generate-mcqs/', GenerateMCQs.as_view(), name='generate-mcqs'),
    path('my-mcq-requests/', UserMCQRequestsView.as_view(), name='my-mcq-requests'), # New URL for retrieving requests
]