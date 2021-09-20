from rest_framework import serializers
from .models import Predictdata, UserInfo, ParentMatchPatientModel, User_data
from django.contrib.auth.models import User
from rest_framework.authtoken.views import Token

class PredictdataSerializer(serializers.ModelSerializer):
    class Meta:
        model = Predictdata
        #fields = ['id','user_id','name','predict','datetime']
        fields = ['id','user','predict','datetime']
    """docstring for ."""

    #def __init__(self, arg):
    #    super(, self).__init__()
    #    self.arg = arg
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'email']

        #讓正常情況下回傳的response看不到password
        extra_kwargs = {'password':{
            'write_only':True,
            'required':True
        }}

    #讓password是hash過的
    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        Token.objects.create(user = user)
        return user

class UserInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserInfo
        #fields = ['id','user_id','name','predict','datetime']
        fields = ['id','user','name','sex','height','weight','birthday','parentName','parentPhone']

#新個人資料modelserialix
class UserDataSerializer(serializers.ModelSerializer):
   class Meta:
      model = User_data
      fields = [
                'user',
                'account',
                'user_name',
                'height',
                'weight',
                'sexual',
                'birth_date',
                'contactor',
                'contactor_phone'
                ]

class ParentMatchPatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParentMatchPatientModel
        #fields = ['id','user_id','name','predict','datetime']
        fields = ['id','parent_id','patient_id']
