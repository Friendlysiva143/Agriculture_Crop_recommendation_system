"""
URL Configuration for Predictions App
"""

from django.urls import path
from . import views

app_name = 'predictions'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('download/', views.download_results, name='download_results'),
    path('history/', views.history, name='history'),
]