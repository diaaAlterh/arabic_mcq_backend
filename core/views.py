import logging # Import for logging errors

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

from core.logic.crew_engine import run_mcq_pipeline
from .models import MCQRequest # Import the new model
from .serializers import MCQRequestSerializer # Import the new serializer

logger = logging.getLogger(__name__) # Initialize logger

class GenerateMCQs(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        text = request.data.get('text')
        if not text:
            return Response({"error": "Missing 'text' in request"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 1. Run the MCQ generation pipeline
            result = run_mcq_pipeline(text)

            # 2. Save the request and its result to the database
            # request.user is available because of JWTAuthentication and IsAuthenticated
            mcq_request = MCQRequest.objects.create(
                user=request.user,
                input_text=text,
                generated_mcqs=result # Stores the JSON response from the pipeline
            )

            # 3. Return the generated MCQs to the user
            
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            # Log the full traceback for debugging purposes
            logger.exception("Error occurred while generating MCQs")  # This logs full traceback
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