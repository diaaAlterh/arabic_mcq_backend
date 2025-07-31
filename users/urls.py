from django.urls import path

from .views import RegisterView, LoginView

from rest_framework_simplejwt.views import (
    TokenRefreshView,
    # TokenObtainPairView, # Only uncomment if you decide to use simplejwt's default login view
)


urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'), # Your custom login view
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

]