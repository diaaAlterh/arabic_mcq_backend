from django.urls import path
from core.views.mcq_view import GenerateMCQs

urlpatterns = [
    path('generate-mcqs/', GenerateMCQs.as_view(), name='generate-mcqs'),
]
