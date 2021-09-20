from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
import numpy as np
import joblib
from rest_framework.decorators import api_view, APIView
from rest_framework.response import Response
from .models import Predictdata, UserInfo, ParentMatchPatientModel, User_data
from .serializers import PredictdataSerializer, UserSerializer, UserInfoSerializer, ParentMatchPatientSerializer, UserDataSerializer
from rest_framework import status
from rest_framework import generics
from rest_framework import mixins
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authtoken.views import Token
from django.contrib.auth.models import User
from rest_framework.authentication import TokenAuthentication
from django.contrib.auth import authenticate
from rest_framework.authtoken.views import ObtainAuthToken
import statistics
import scipy.stats
import math

# Create your views here.
#def index(request):
#    if request.method == "POST":
#        return HttpResponse("It is working")

#    elif request.method == "GET":
#        return HttpResponse("It is GET")
"""
@api_view(['GET'])
def test(request):
    #return_data = {
    #    "data" : "1",
    #    "message" : "Successful",
    #}
    predict_data = [3.5,1,0.01,4.0]
    sc = joblib.load('model/knn_scaled_model.pkl', mmap_mode ='r')
    predict_data = sc.transform([predict_data])
    #Passing data to model & loading the model from disks
    model_path = 'model/knn_model.pkl'
    classifier = joblib.load('model/knn_model.pkl', mmap_mode ='r')
    prediction = classifier.predict(predict_data)
    #conf_score =  np.max(classifier.predict_proba([result]))*100
    if prediction == 0:
        result = "run"
    else:
        result = "walk"
    predictions = {
        #'error' : '0',
        #'message' : 'Successfull',
        #'prediction' : prediction,
        #'confidence_score' : conf_score
        "predict_result": result
    }
    return Response(predictions)

@api_view(["POST"])
def predict_stroke(request):
    try:
        nlp = request.data.get('nlp',None)
        p2p = request.data.get('p2p',None)
        rms = request.data.get('rms',None)
        auc = request.data.get('auc',None)

        fields = [nlp,p2p,rms,auc]
        print(fields)
        if not None in fields:
            #Datapreprocessing Convert the values to float
            nlp = float(nlp)
            p2p = float(p2p)
            rms = float(rms)
            auc = float(auc)

            predict_data = [nlp,p2p,rms,auc]
            sc = joblib.load('model/knn_scaled_model.pkl', mmap_mode ='r')
            predict_data = sc.transform([predict_data])
            print(predict_data)
            #Passing data to model & loading the model from disks
            model_path = 'model/knn_model.pkl'
            classifier = joblib.load('model/knn_model.pkl', mmap_mode ='r')
            prediction = classifier.predict(predict_data)
            #conf_score =  np.max(classifier.predict_proba([result]))*100
            if prediction == 0:
                result = "run"
            else:
                result = "walk"
            predictions = {
                #'error' : '0',
                #'message' : 'Successfull',
                #'prediction' : prediction,
                #'confidence_score' : conf_score
                "predict_result": result
            }
        else:
            predictions = {
                'error' : '1',
                'message': 'Invalid Parameters',
                'predict_result': 'fail'
            }
    except Exception as e:
        predictions = {
            'error' : '2',
            "message": str(e),
            'predict_result': str(e)
        }

    return Response(predictions)
"""

class PredictStroke(APIView):
    authentication_classes = [TokenAuthentication]
    #premissions_classes = [AllowAny]

    def post(self, request):
        try:
            nlp = request.data.get('nlp',None)
            p2p = request.data.get('p2p',None)
            rms = request.data.get('rms',None)
            auc = request.data.get('auc',None)
            user_id = request.data.get('user_id',None)

            fields = [nlp,p2p,rms,auc]
            print(fields)
            if not None in fields:
                #Datapreprocessing Convert the values to float
                nlp = float(nlp)
                p2p = float(p2p)
                rms = float(rms)
                auc = float(auc)

                predict_data = [nlp,p2p,rms,auc]
                sc = joblib.load('model/knn_scaled_model.pkl', mmap_mode ='r')
                predict_data = sc.transform([predict_data])
                print(predict_data)
                #Passing data to model & loading the model from disks
                model_path = 'model/knn_model.pkl'
                classifier = joblib.load('model/knn_model.pkl', mmap_mode ='r')
                prediction = classifier.predict(predict_data)
                #conf_score =  np.max(classifier.predict_proba([result]))*100
                if prediction == 0:
                    result = "run"
                else:
                    result = "walk"
                response = {
                    #'error' : '0',
                    #'message' : 'Successfull',
                    #'prediction' : prediction,
                    #'confidence_score' : conf_score
                    "user": user_id,
                    "predict": result,

                }
                serializer = PredictdataSerializer(data = response)

                #檢查post來的資料格式跟資料庫的有沒有一樣
                if serializer.is_valid():
                    #print(serializer.data)
                    serializer.save()
                    #return Response(serializer.data, status = status.HTTP_201_CREATED)
                    return Response(response, status = status.HTTP_201_CREATED)
                return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
            else:
                response = {
                    'error' : '1',
                    'message': 'Invalid Parameters',
                    'predict': 'fail'
                }
                return Response(response)
        except Exception as e:

            response = {
                'error' : '2',
                "message": str(e),
                'predict': str(e)
            }
            return Response(response)
    def get_object(self, user_id):
        try:
            return Predictdata.objects.get(user=user_id)

        except Predictdata.DoesNotExist:
            return Response(status = status.HTTP_404_NOT_FOUND)

    def get(self, request):
        predict = self.get_object(request.data.get("user_id"))
        serializer = PredictdataSerializer(predict)
        return Response(serializer.data)

    def put(self, request):
        predict = self.get_object(request.data.get("user_id"))
        serializer = PredictdataSerializer(predict, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        predict = self.get_object(request.data.get("user_id"))
        predict.delete()
        return Response(status = status.HTTP_204_NO_CONTENT)




"""
class PredictByUserId(APIView):
    authentication_classes = [TokenAuthentication]
    #authentication_classes = []
    premission_classes = [IsAuthenticated]
    #permission_classes = [AllowAny]
    #permission_classes = []



    def get(self, request):
        predicts = Predictdata.objects.all()
        serializer = PredictdataSerializer(predicts, many = True)
        return Response(serializer.data)

    request.data是dictionary型態

    def get_object(self, user_id):
        try:
            return Predictdata.objects.get(user=user_id)

        except Predictdata.DoesNotExist:
            return Response(status = status.HTTP_404_NOT_FOUND)

    def get(self, request, user_id):
        predict = self.get_object(user_id)
        serializer = PredictdataSerializer(predict)
        return Response(serializer.data)

    def put(self, request, user_id):
        predict = self.get_object(user_id)
        serializer = PredictdataSerializer(predict, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

    def delete(self, request, user_id):
        predict = self.get_object(user_id)
        predict.delete()
        return Response(status = status.HTTP_204_NO_CONTENT)
"""
"""
    def post(self, request, user_id):
        predict = self.get_object(user_id)
        serializer = PredictdataSerializer(predict, data = request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
"""


"""
    def post(self, request):
        serializer = PredictdataSerializer(data = request.data)
        #檢查post來的資料格式跟資料庫的有沒有一樣
        if serializer.is_validation:
            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)
"""
# for user register
class UserRegister(APIView):
    #queryset = User.objects.all()
    #serializer = UserSerializer()
    authentication_classes = []
    permission_classes = [AllowAny]
    #authentication_classes = [TokenAuthentication]
    #premissions_classes = [IsAuthenticated]

    def post(self, request):
        serializer = UserSerializer(data = request.data)

        #檢查post來的資料格式跟資料庫的有沒有一樣
        if serializer.is_valid():
            #if User.objects.get(username=serializer.data.username) != None:
                #print(serializer.errors)
                #return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)
        #print(serializer.errors)
        """假如已註冊過此username時要回傳 username=registered 讓Android端知道，讓user重註冊一次"""
        if User.objects.get(username=serializer.data.get('username')) !=None:
            return Response({"username": "registered"}, status = status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

class UserInfoManage(APIView):
    authentication_classes = [TokenAuthentication]
    #authentication_classes = []
    premission_classes = [IsAuthenticated]
    #permission_classes = [AllowAny]
    #permission_classes = []


    """
    def get(self, request):
        predicts = Predictdata.objects.all()
        serializer = PredictdataSerializer(predicts, many = True)
        return Response(serializer.data)

    request.data是dictionary型態
    """
    def get_object(self, user_id):
        try:
            return UserInfo.objects.get(user=user_id)

        except UserInfo.DoesNotExist:
            return Response(status = status.HTTP_404_NOT_FOUND)

    def get(self, request):
        userinfo = self.get_object(request.data.get("user_id"))
        serializer = UserInfoSerializer(userinfo)
        return Response(serializer.data)

    def put(self, request):
        userinfo = self.get_object(request.data.get("user_id"))
        serializer = UserInfoSerializer(userinfo, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        userinfo = self.get_object(request.data.get("user_id"))
        userinfo.delete()
        return Response(status = status.HTTP_204_NO_CONTENT)

    def post(self, request):
        serializer = UserInfoSerializer(data = request.data)

        #檢查post來的資料格式跟資料庫的有沒有一樣
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)

        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

#新個人資料APIView
class UserData(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get_data(self, user_id):

            try:
                return User_data.objects.get(user=user_id)

            except User_data.DoesNotExist:
                """
                userdata = User_data()
                userdata.id = 0
                userdata.user = 0
                userdata.account = "nan"
                userdata.user_name = "nan"
                userdata.height = "nan"
                userdata.weight = "nan"
                userdata.sexual = "nan"
                userdata.birth_date = "nan"
                userdata.contactor = "nan"
                userdata.contactor_phone = "nan"

                return userdata
                """
                return Response({'status': 'Profile does not build .'},safe=False)

    def get(self,request, user_id):
        try:
            one_data = self.get_data(user_id)
            serializer = UserDataSerializer(one_data)
            return Response(serializer.data)
            # return JsonResponse(str(serializer.data),safe=False)
            #return HttpResponse(json.dumps(serializer.data), content_type = "application/json")
        except:
            return Response({'exist':'no'})

    def post (self,request, user_id):
        try:
            User_data.objects.create(
                                    user = user_id,
                                    account = request.data.get("email")
                                    )
            #return JsonResponse({'status': 'ok, This is built.'})
            return Response({'status': 'ok, This is built.'})
        except:
            print(str(request))
            #return JsonResponse({'status':'post is failed','data':str(request.data)})
            return Response({'status':'post is failed','data':str(request.data)})
    def put(self, request, user_id):
        one_data = self.get_data(user_id)
        serializer = UserDataSerializer(one_data,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, safe=False)
        return JsonResponse(serializer.errors, safe=False)

"""登入的APIView"""
class CustomAuthToken(ObtainAuthToken):

    authentication_classes=[]
    permission_classes=[AllowAny]

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        #有這個user
        if user is not None:
            serializer = self.serializer_class(data=request.data,context={'request': request})
            serializer.is_valid(raise_exception=True)
            user = serializer.validated_data['user']
            token = Token.objects.get(user=user)
            return Response({
                'token': token.key,
                'user_id': user.pk,
                'username': user.username,
                'email': user.email
            })
        #沒這個user
        else:
            return Response({
                'token': "unregistered",
                'user_id': "unregistered",
                'username': "unregistered",
                'email': "unregistered"
            })

class ParentMatchPatient(APIView):
    authentication_classes = [TokenAuthentication]
    #authentication_classes = []
    premission_classes = [IsAuthenticated]
    #permission_classes = [AllowAny]
    #permission_classes = []


    """
    def get(self, request):
        predicts = Predictdata.objects.all()
        serializer = PredictdataSerializer(predicts, many = True)
        return Response(serializer.data)

    request.data是dictionary型態
    """
    def get_object(self, parent_id):
        try:
            return ParentMatchPatientModel.objects.get(parent_id = parent_id)

        except ParentMatchPatientModel.DoesNotExist:
            #return Response(status = status.HTTP_404_NOT_FOUND)
            p = ParentMatchPatientModel()
            p.id = 0
            p.parent_id = 0
            p.patient_id = 0
            return p

    def get(self, request):
        parentmatchpatient = self.get_object(request.data.get("parent_id"))
        serializer = ParentMatchPatientSerializer(parentmatchpatient)
        return Response(serializer.data)

    def put(self, request):
        parentmatchpatient = self.get_object(request.data.get("parent_id"))
        serializer = ParentMatchPatientSerializer(parentmatchpatient, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        parentmatchpatient = self.get_object(request.data.get("parent_id"))
        parentmatchpatient.delete()
        return Response(status = status.HTTP_204_NO_CONTENT)

    def post(self, request):
        serializer = ParentMatchPatientSerializer(data = request.data)

        #檢查post來的資料格式跟資料庫的有沒有一樣
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = status.HTTP_201_CREATED)

        return Response(serializer.errors, status = status.HTTP_400_BAD_REQUEST)


class DeliverApi(APIView):
    authentication_classes = []

    permission_classes = [AllowAny]

    def post(self, request):
        try:
            #右手
            right_list = []
            right_window_dic = request.data.get("right")

            for n in right_window_dic.values():
                right_list.append(float(n))

            print("right_list")
            print(right_list)
            #左手
            left_list = []
            left_window_dic = request.data.get("left")

            for n in left_window_dic.values():
                left_list.append(float(n))
            print("left_list")
            print(left_list)

            """abnormal intensity"""

            #判斷兩手每分鐘有無動作(有動作值不變，無動作值變0)
            #右手
            i = 0
            r_intensity_window = []
            while i<len(right_list):
                r_intensity_window = r_intensity_window + self.motion_detector(right_list[i:i+60])
                i = i+60
            print("r_intensity_window")
            print(r_intensity_window)
            #左手
            i = 0
            l_intensity_window = []
            while i<len(left_list):
                l_intensity_window = l_intensity_window + self.motion_detector(left_list[i:i+60])
                i = i+60
            print("l_intensity_window")
            print(l_intensity_window)
            #特徵擷取(算統計值)(還沒加之前的4個)
            right_feature = self.feature(r_intensity_window)
            print("right_feature")
            print(right_feature)

            left_feature = self.feature(l_intensity_window)
            print("left_feature")
            print(left_feature)
            predict_intensity_list = self.hl(right_feature, left_feature)
            print("predict_intensity_list")
            print(predict_intensity_list)

            #min max transformation

            #丟入訓練好的abnormal intensity的模型

            """abnormal frequency"""

            #每分鐘判斷有無動作(有動作:true 無動作:false)
            #右手
            i = 0
            r_frequency_window = []
            while i<len(right_list):
                r_frequency_window.append(self.check_motion_per_min(right_list[i:i+60]))
                i = i+60
            print("r_frequency_window")
            print(r_frequency_window)
            #左手
            i = 0
            l_frequency_window = []
            while i<len(left_list):
                l_frequency_window.append(self.check_motion_per_min(left_list[i:i+60]))
                i = i+60
            print("l_frequency_window")
            print(l_frequency_window)
            ##兩隻手都有動作:0 一隻手或兩隻手都沒動作:1
            predict_frequency_list = self.swmd(r_frequency_window, l_frequency_window)
            print("predict_frequency_list")
            print(predict_frequency_list)

            #丟入訓練好的abnormal frequency模型

            #拿intensity和frequency的模型預測結果一起判斷(其中一個預測結果是中風就判斷為中風)

            response = {

                "result": "success",

            }
            return Response(response)
        except Exception as e:
            print(str(e))
            return Response({"result": str(e)})

    #統計值特徵
    def feature(self,hand_motion_list):
        hl_feature = []

        # average
        avg = np.mean(hand_motion_list)
        hl_feature.append(avg)
        #standard deviation
        std = statistics.stdev(hand_motion_list)
        hl_feature.append(std)
        #variation
        var = np.var(hand_motion_list)
        hl_feature.append(var)
        #median
        sortedLst = sorted(hand_motion_list)
        lstLen = len(hand_motion_list)
        index = (lstLen - 1) // 2

        if (lstLen % 2):
            hl_feature.append(sortedLst[index])
        else:
            hl_feature.append((sortedLst[index] + sortedLst[index + 1])/2.0)

        #mean absolute deviation
        s = 0
        for val in hand_motion_list:
            s+=abs(val-avg)
        mad = s/len(hand_motion_list)
        hl_feature.append(mad)

        #max
        maximum = max(hand_motion_list)
        hl_feature.append(maximum)

        #sum
        total = sum(hand_motion_list)
        hl_feature.append(total)

        #root mean square
        s = 0
        for val in hand_motion_list:
            s+=val**2
        rms = (s/len(hand_motion_list))**0.5
        hl_feature.append(rms)

        #Quartile 25%
        q25 = np.percentile(hand_motion_list, 25)
        hl_feature.append(q25)

        #Quartile 75%
        q75 = np.percentile(hand_motion_list, 75)
        hl_feature.append(q75)

        #Inter Quartile Range(q75 - q25)
        iqr = q75 - q25
        hl_feature.append(iqr)

        #Sample Skewness
        skewness = scipy.stats.skew(hand_motion_list)
        hl_feature.append(skewness)
        #Sample Kurtosis
        kurtosis = scipy.stats.kurtosis(hand_motion_list)
        hl_feature.append(kurtosis)


        return hl_feature

    #hemiparesis likelihood
    def hl(self,right, left):
        hl_list = []
        #兩隻手動作不能比較HL就設0
        for rf, lf in zip(right, left):
            try:
                if ~np.isfinite(rf/lf):
                    hl_list.append(0)
                elif rf==0 or lf==0:
                    hl_list.append(0)
                else:
                    likelihood = abs(math.log(rf/lf,2))
                    hl_list.append(likelihood)
            except:
                hl_list.append(0)

        return hl_list

    #threshold(0.045)判斷每分鐘有無動作，有動作加速度motion不變，沒動作加速度motion=0
    def motion_detector(self,motion_per_min):
    #return ((a-9.8)**2).mean()
        motion_list = []

        #有動作時這一分鐘的加速度跟原本一樣
        sum_per_min = 0
        for m in motion_per_min:
            sum_per_min = sum_per_min + ((m-9.8)**2)
        if (sum_per_min/len(motion_per_min))>=0.045:
            motion_list=motion_per_min
            #沒動作時這一分鐘的加速度設為0
        else:
            motion_list=[0]*len(motion_per_min)

        return motion_list

    #每分鐘判斷有無動作
    def check_motion_per_min(self,motion_per_min):

        sum_per_min = 0
        for m in motion_per_min:
            sum_per_min = sum_per_min + ((m-9.8)**2)
        #有動作:true
        if (sum_per_min/len(motion_per_min))>=0.045:
            return True
        #沒動作:false
        else:
            return False

    #每分鐘兩隻手都有動作:0 一隻手或兩隻手都沒動作:1
    def swmd(self,rm,lm):
        window_list = []
        #左右兩手的window數量一樣(兩手手環同步時)
        if len(rm) == len(lm):

            for  rm_min_segment,lm_min_segment in zip(rm, lm):
                #兩手都有動作:0
                if rm_min_segment and lm_min_segment:
                    window_list.append(0)
                #一手或兩手沒動作:1
                else:
                    window_list.append(1)

        #左右兩手的window數量不一樣(兩手手環不同步)(應該不會發生)
        else:
            #選左右兩手window數量最少的
            shortest_len = min(len(rm),len(lm))

            for  rm_min_segment,lm_min_segment in zip(rm[0:shortest_len],lm[0:shortest_len]):
                #兩手都有動作:0
                if rm_min_segment and lm_min_segment:
                    window_list.append(0)
                #一手或兩手沒動作:1
                else:
                    window_list.append(1)
            window_list = window_list + [0]*(120 - len(window_list))

        return window_list
