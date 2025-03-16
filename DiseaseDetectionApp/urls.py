from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('detect_disease/',views.detect_disease,name='detection'),
]
