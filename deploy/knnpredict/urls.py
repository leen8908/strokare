
from django.urls import path
from .views import PredictStroke, UserRegister, UserInfoManage, CustomAuthToken, ParentMatchPatient, DeliverApi, UserData
from rest_framework.routers import DefaultRouter


router = DefaultRouter()
#router.register(r'articles', ArticleViewSet, basename='article')
#urlpatterns = router.urls


urlpatterns = [
    #path('', test),
    #path('predict', predict_stroke),
    path('predict', PredictStroke.as_view()),
    #path('predict/<int:user_id>', PredictByUserId.as_view()),
    path('register', UserRegister.as_view()),
    path('userinfo', UserInfoManage.as_view()),
    path('login', CustomAuthToken.as_view()),
    path('parent_id_match_patient_id', ParentMatchPatient.as_view()),
    path('deliver', DeliverApi.as_view()),
    path('userdata/<int:user_id>/', UserData.as_view()),
]
