from rest_framework import serializers
from .models import MCQRequest
from .models import Task

class MCQRequestSerializer(serializers.ModelSerializer):
    # Optional: If you want to include the username instead of just the user ID
    # user = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = MCQRequest
        fields = ['id', 'user', 'input_text', 'generated_mcqs', 'created_at']
        # 'user' and 'created_at' are automatically handled by the model
        # 'id' is automatically generated
        read_only_fields = ['user', 'created_at'] # These fields are set by the system, not directly by user input
        
class TextInputSerializer(serializers.Serializer):
    text = serializers.CharField(min_length=50)
    question_count = serializers.IntegerField(default=5, min_value=1, max_value=50)

class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Task
        fields = "__all__"