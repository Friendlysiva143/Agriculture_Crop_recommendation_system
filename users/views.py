"""
User authentication views
"""

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.decorators.http import require_http_methods

from .forms import RegisterForm, LoginForm

@require_http_methods(["GET", "POST"])
def register(request):
    """
    User registration view
    """
    if request.user.is_authenticated:
        return redirect('predictions:dashboard')
    
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Account created successfully! Please login.')
            return redirect('users:login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = RegisterForm()
    
    context = {'form': form}
    return render(request, 'users/register.html', context)


@require_http_methods(["GET", "POST"])
def login_view(request):
    """
    User login view
    """
    if request.user.is_authenticated:
        return redirect('predictions:dashboard')
    
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username_or_email = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            # Try to authenticate with username first
            user = authenticate(request, username=username_or_email, password=password)
            
            # If not found, try with email
            if not user:
                try:
                    user_by_email = User.objects.get(email=username_or_email)
                    user = authenticate(request, username=user_by_email.username, password=password)
                except User.DoesNotExist:
                    user = None
            
            if user:
                login(request, user)
                messages.success(request, f'Welcome back, {user.first_name or user.username}!')
                return redirect('predictions:dashboard')
            else:
                messages.error(request, 'Invalid username/email or password.')
    else:
        form = LoginForm()
    
    context = {'form': form}
    return render(request, 'users/login.html', context)


def logout_view(request):
    """
    User logout view
    """
    logout(request)
    messages.success(request, 'Logged out successfully!')
    return redirect('users:login')
