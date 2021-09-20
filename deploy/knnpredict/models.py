from django.db import models


# Create your models here.
class Predictdata(models.Model):
    user = models.IntegerField()
    #name = models.CharField(max_length = 100)
    predict = models.CharField(max_length = 100)
    datetime = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.user)

class UserInfo(models.Model):
    user = models.IntegerField() #user_id
    name = models.CharField(max_length = 100)
    sex = models.CharField(max_length = 100)
    height = models.FloatField()
    weight = models.FloatField()
    birthday = models.DateField(auto_now = False,auto_now_add=False)
    parentName = models.CharField(max_length=100)
    parentPhone = models.CharField(max_length=100)
    def __str__(self):
        return self.name

class ParentMatchPatientModel(models.Model):
    parent_id = models.IntegerField()
    patient_id = models.IntegerField()


    def __str__(self):
        return str(self.parent_id)
#新個人資料model
class User_data(models.Model):

    # user = models.ForeignKey( Account, on_delete=models.CASCADE )
    user = models.CharField(max_length=10,default='')
    account = models.CharField(max_length=100, default='example@gmail.com')
    user_name = models.CharField(max_length=100,default='NAN')
    height = models.CharField(max_length=20,default='NAN')
    weight = models.CharField(max_length=20,default='NAN')
    sexual = models.CharField(max_length=100,default='NAN')
    birth_date = models.CharField(max_length=100,default='NAN')
    contactor = models.CharField(max_length=100 ,default='NAN')
    contactor_phone = models.CharField(max_length=100,default=9999999999)

    class Meta:
       db_table ='user_data_model'
