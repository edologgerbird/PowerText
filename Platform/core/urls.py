from django.urls import path

from . import views

urlpatterns = [
    path('/demo1/', views.user_demo_1, name='user_demo_1'),
    path('/demo2/', views.user_demo_2, name='user_demo_2')
]
