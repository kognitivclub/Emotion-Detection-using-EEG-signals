from django.shortcuts import render
from .svm_model import emotion
import json
def home(request):
    context={}
    return render(request,"home.html",context)
def predict(request):
    x1=request.POST['valence']
    x2=request.POST['arousal']
    x3=request.POST['dominance']
    x4=request.POST['liking']
    x5=request.POST['familiarity']
    x6=request.POST['relevance']
    result=emotion(x1,x2,x3,x4,x5,x6)
    return render(request,"home.html",{"emotion":result[0]})