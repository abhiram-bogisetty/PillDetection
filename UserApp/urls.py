
from django.urls import path
from UserApp import views
urlpatterns = [
    path('Register', views.Register),
    path('index', views.index),
    path('RegAction', views.RegAction),
    path('LogAction', views.LogAction),
    path('home', views.home),
    path('ModelGraphs',views.ModelGraphs),
    path('Profile',views.Profile),
    path('Upload', views.Upload),
    path('imageAction', views.imageAction),
    path('Test', views.Test),
    path('logout', views.logout)

]
