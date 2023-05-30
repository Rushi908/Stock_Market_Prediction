from django.db import models


class User(models.Model):
    username = models.TextField(primary_key=True)
    password = models.TextField()
    role = models.TextField()

    class Meta:
        db_table = 'tblUser'


class Company(models.Model):
    name = models.TextField(primary_key=True)
    ticker = models.TextField()

    class Meta:
        db_table = 'tblCompany'
