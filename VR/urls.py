"""VR URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.conf.urls.static import static
from django.urls import path
import app.views
from VR import settings

urlpatterns = [
                  path('', app.views.login),
                  path('register', app.views.register),
                  path('addcompany', app.views.add_company),
                  path('deletecompany', app.views.delete_company),
                  path('uploaddataset', app.views.upload_dataset),
                  path('analysis', app.views.analysis),
                  path('searchcompany', app.views.search_company),
                  path('prediction', app.views.prediction),

              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
