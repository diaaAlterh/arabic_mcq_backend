from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.logic.crew_engine import run_mcq_pipeline

class GenerateMCQs(APIView):
    def post(self, request):
        text = request.data.get('text')
        if not text:
            return Response({"error": "Missing 'text' in request"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = run_mcq_pipeline(text)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
