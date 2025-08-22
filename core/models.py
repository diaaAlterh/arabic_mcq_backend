from django.db import models
from django.conf import settings # To get the AUTH_USER_MODEL
import uuid


class MCQRequest(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, # Links to your custom User model
        on_delete=models.CASCADE,
        related_name='mcq_requests',
        help_text="The user who generated this MCQ set."
    )
    input_text = models.TextField(
        help_text="The original text provided by the user for MCQ generation."
    )
    generated_mcqs = models.JSONField(
        help_text="The JSON output containing the generated MCQs."
    )
    created_at = models.DateTimeField(
        auto_now_add=True, # Automatically sets the timestamp when the object is first created
        help_text="Timestamp of when the MCQ request was made."
    )

    class Meta:
        # Orders results by most recent first when querying
        ordering = ['-created_at']
        verbose_name = "MCQ Request"
        verbose_name_plural = "MCQ Requests"

    def __str__(self):
        return f"Request by {self.user.username} on {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
class Task(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    progress = models.IntegerField(default=0)
    result = models.JSONField(null=True, blank=True)
    error = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Task {self.id} - {self.status}"
