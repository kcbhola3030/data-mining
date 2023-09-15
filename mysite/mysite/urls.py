"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from . import views



urlpatterns = [ 
    path("admin", admin.site.urls),
    path('calculate', views.upload_csv, name='upload_csv' ), 
    path('pearson', views.calculate_pearson, name='pearson' ), 
    path('classify/<str:method>/', views.classify, name='classify'),
    path('regression/', views.regression_classifier, name='regression_classifier'),
    path('naive_bayesian/', views.naive_bayesian_classifier, name='naive_bayesian_classifier'),
    path('knn/', views.knn_classifier, name='knn_classifier'),
    path('ann/', views.ann_classifier, name='ann_classifier'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 
