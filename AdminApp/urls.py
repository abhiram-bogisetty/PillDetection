
from django.urls import path
from AdminApp import views
urlpatterns = [
    path('', views.index),
    path('AdminAction',views.AdminAction),
    path('Adminhome', views.Adminhome),
    path('ViewAllUsers', views.ViewAllUsers),
    path('Delete', views.Delete),
    path('UploadDataset', views.UploadDataset),
    path('DataGenerate', views.DataGenerate),
    path('GenerateCNN', views.GenerateCNN),
    path('logout', views.logout)
]
