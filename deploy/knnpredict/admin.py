from django.contrib import admin
from .models import Predictdata, UserInfo, ParentMatchPatientModel, User_data

# Register your models here.
#admin.site.register(Userdata)

@admin.register(Predictdata)
class PredictdataModel(admin.ModelAdmin):
    list_filter = ('user', 'predict')
    list_display = ('user', 'predict')

@admin.register(UserInfo)
class UserInfoModel(admin.ModelAdmin):
    list_filter = ('id', 'name')
    list_display = ('id', 'name')

@admin.register(ParentMatchPatientModel)
class ParentPatientModel(admin.ModelAdmin):
    list_filter = ('id', 'parent_id')
    list_display = ('id', 'parent_id')

@admin.register(User_data)
class UserDataModel(admin.ModelAdmin):
    list_filter = ('id', 'user')
    list_display = ('id', 'user')
