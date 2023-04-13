import random

import numpy as np
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .handler_bert import *


def homepage(request):
    return render(request, "index.html")


@csrf_exempt
def user_demo_1(request):
    context = {}
    if request.method == 'POST':
        if request.POST['form_id'] == 'text_submit':
            try:
                text = request.POST['comment']
                predict = wrapper_hate_bert(text)

                # render text output: possible implementation
                context['displayed_text'] = shrink_text(text)
                flagged = '0'
                violations = []
                if predict[0] == 1:
                    flagged = '1'
                    violations.append('hate')
                if predict[1] == 1:
                    flagged = '1'
                    violations.append('privacy')
                if predict[2] == 1:
                    flagged = '1'
                    violations.append('sexual')
                if predict[3] == 1:
                    flagged = '1'
                    violations.append('impersonation')
                if predict[4] == 1:
                    flagged = '1'
                    violations.append('illegal')
                if predict[5] == 1:
                    flagged = '1'
                    violations.append('advertisement')
                if predict[6] == 1:
                    flagged = '1'
                    violations.append('AI generated')

                if flagged == '0':
                    message = 'The post passed text screening.'
                else:
                    message = 'Detected! Possible '
                    for i in violations:
                        message = message + i + ", "
                    message = message[:-2] + " content!"
                context['text_flagged'] = flagged
                context['text_message'] = message
            except Exception:
                pass

    return render(request, "user_demo_1.html", context)


@csrf_exempt
def user_demo_2(request):
    context = {}
    if request.method == 'POST':
        if request.POST['form_id'] == 'text_submit':
            try:
                text = request.POST['comment']
                predict = wrapper_hate_bert(text)

                # render labels: possible implementation
                context['text_submitted'] = text
                context['label_hate'] = '1' if predict[0] == 1 else '0'
                context['label_private'] = '1' if predict[1] == 1 else '0'
                context['label_sexual'] = '1' if predict[2] == 1 else '0'
                context['label_imperson'] = '1' if predict[3] == 1 else '0'
                context['label_illegal'] = '1' if predict[4] == 1 else '0'
                context['label_ads'] = '1' if predict[5] == 1 else '0'
                context['label_ai'] = '1' if predict[6] == 1 else '0'
            except Exception:
                pass

    return render(request, "user_demo_2.html", context)

# helper functions
def wrapper_all_zero(text):
    return np.zeros(7)

def wrapper_random_guess(text):
    output = []
    for i in range(7):
        output.append(random.choice([0, 1]))
    return output

def wrapper_hate_bert(text):
    predict = run_hate_bert(text)
    print(predict)
    output = []
    threshold = 6
    for i in predict:
        if i >= threshold:
            output.append(1)
        else:
            output.append(0)
    print(output)
    return output

def shrink_text(text):
    if len(text) >= 30:
        text = text[:12] + "..." + text[-12:]
    return text