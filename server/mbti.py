class PersonalitySurvey:
    def __init__(self):
        self.social_type = None  # S for Sociable, F for Family-centered/Romantic
        self.cultural_preference = (
            None  # A for Art/Culture-centered, N for Nature/History-centered
        )
        self.diet_type = (
            None  # H for Health/Weight-centered, F for Natural/Flexibility-centered
        )
        self.fruit_preference = None  # C for Classic/Luxurious, S for Sweet/Unique
        self.hobby_type = None  # I for Balance/Cook, E : Gaming/fashion

    def question_answer(self, answers):
        # 불금을 보내는 방법
        if answers[1] in ["칵테일 바", "캠핑이나 바닷가"]:
            self.social_type = "E"
        else:
            self.social_type = "I"

        # 어느 나라에서 살고 싶은지
        if answers[2] in ["호주", "이탈리아"]:
            self.cultural_preference = "N"
        else:
            self.cultural_preference = "A"

        # 어떤 식단을 따르는지
        if answers[3] in ["키토 식단", "글루텐-프리 식단"]:
            self.diet_type = "H"
        else:
            self.diet_type = "F"

        # 가장 좋아하는 과일
        if answers[4] in ["사과", "체리"]:
            self.fruit_preference = "C"
        else:
            self.fruit_preference = "S"

        # 당신의 취미
        if answers[5] in ["요가", "요리"]:
            self.hobby_type = "E"
        else:
            self.hobby_type = "I"

    def personality_type(self):
        return (
            self.social_type
            + self.cultural_preference
            + self.diet_type
            + self.fruit_preference
            + self.hobby_type
        )


# 사용 예제
survey = PersonalitySurvey()

# 설문 응답을 추가합니다.
answers = {1: "칵테일 바", 2: "이탈리아", 3: "키토 식단", 4: "사과", 5: "게임"}
survey.question_answer(answers)

# 개인의 성격 유형을 출력합니다.
print(survey.personality_type())


def get_survey_choices(answers):
    survey_choices = []

    if 1 in answers:
        if answers[1] == "칵테일 바":
            survey_choices.extend(["칵테일 바", "음료", "시간", "즐거운 시간"])
        elif answers[1] == "가족":
            survey_choices.extend(["가족", "저녁", "대화", "웃음", "행복한 시간"])
        elif answers[1] == "데이트":
            survey_choices.extend(["데이트", "연인", "로맨틱한 분위기", "서로를 알아가는 시간"])
        elif answers[1] == "캠핑":
            survey_choices.extend(["캠핑", "바닷가", "음식", "야외 분위기", "활기찬 시간"])

    if 2 in answers:
        if answers[2] == "이탈리아":
            survey_choices.extend(["풍부한 음식", "예술"])
        elif answers[2] == "프랑스":
            survey_choices.extend(["아름다운 도시", "문화"])
        elif answers[2] == "호주":
            survey_choices.extend(["푸른 자연", "쾌적한 환경"])
        elif answers[2] == "스페인":
            survey_choices.extend(["열정적인 문화", "아름다운 해변", "여유로운 삶"])

    if 3 in answers:
        if answers[3] == "키토 식단":
            survey_choices.extend(["탄수화물 제한", "지방과 단백질 중심"])
        elif answers[3] == "생식 식단":
            survey_choices.extend(["신선한 과일", "채소", "고기", "가공하지 않은 식재료"])
        elif answers[3] == "글루텐-프리 식단":
            survey_choices.extend(["밀가루 피하고 대체 식재료 활용"])
        elif answers[3] == "균형 잡힌 식단":
            survey_choices.extend(["다양한 음식 즐기기"])

    if 4 in answers:
        if answers[4] == "사과":
            survey_choices.extend(["상큼한 맛", "신선한 향"])
        elif answers[4] == "복숭아":
            survey_choices.extend(["달콤하고 쥬시한 식감"])
        elif answers[4] == "체리":
            survey_choices.extend(["산뜻하고 달콤한 맛"])
        elif answers[4] == "라즈베리":
            survey_choices.extend(["상큼하고 과즙이 풍부한 맛"])

    if 5 in answers:
        if answers[5] == "게임":
            survey_choices.extend(["다양한 세계 탐험", "도전"])
        elif answers[5] == "요가":
            survey_choices.extend(["몸과 마음의 균형", "피로 해소"])
        elif answers[5] == "쇼핑":
            survey_choices.extend(["패션 아이템", "다양한 제품 탐색"])
        elif answers[5] == "요리":
            survey_choices.extend(["요리법 익히기", "맛있는 음식 만들기"])

    return survey_choices


answers = {1: "칵테일 바", 2: "이탈리아", 3: "키토 식단", 4: "사과", 5: "게임"}
selections = list(answers.keys())
survey_choices = get_survey_choices(answers)
print(survey_choices)


def get_personality_type(answers):
    personality_type = []

    if 1 in answers:
        if answers[1] == "칵테일 바":
            personality_type.append("사교적이고 외향적인 유형")
        elif answers[1] == "가족":
            personality_type.append("가족 중심적이고 내향적인 유형")
        elif answers[1] == "데이트":
            personality_type.append("로맨틱하고 감성적인 유형")
        elif answers[1] == "캠핑":
            personality_type.append("활동적이고 모험적인 유형")

    if 2 in answers:
        if answers[2] == "프랑스":
            personality_type.append("문화와 예술에 강한 관심이 있는 유형")
        elif answers[2] == "호주":
            personality_type.append("자연과 환경에 중점을 둔 유형")
        elif answers[2] == "이탈리아":
            personality_type.append("음식과 역사를 중요시하는 유형")
        elif answers[2] == "스페인":
            personality_type.append("여유로움과 열정을 동시에 추구하는 유형")

    if 3 in answers:
        if answers[3] == "키토 식단":
            personality_type.append("체중 관리와 건강에 집중하는 유형")
        elif answers[3] == "생식 식단":
            personality_type.append("자연적이고 신선한 음식을 선호하는 유형")
        elif answers[3] == "글루텐-프리 식단":
            personality_type.append("특정한 식습관을 가진, 혹은 특정 식품에 대한 고민을 가진 유형")
        else:
            personality_type.append("유연하고 다양한 음식을 즐기는 유형")

    if 4 in answers:
        if answers[4] == "사과":
            personality_type.append("전통적이고 안정적인 유형")
        elif answers[4] == "복숭아":
            personality_type.append("달콤하고 부드러운 것을 선호하는 유형")
        elif answers[4] == "체리":
            personality_type.append("세련되고 고급스러운 것을 좋아하는 유형")
        elif answers[4] == "라즈베리":
            personality_type.append("독특하고 차별화된 것을 선호하는 유형")

    if 5 in answers:
        if answers[5] == "요가":
            personality_type.append("건강과 내면의 균형을 중요시하는 유형")
        elif answers[5] == "게임":
            personality_type.append("도전과 모험을 즐기는 유형")
        elif answers[5] == "쇼핑":
            personality_type.append("패션과 트렌드에 관심이 많은 유형")
        elif answers[5] == "요리":
            personality_type.append("창조적이고 손재주가 좋은 유형")

    return personality_type


answers = {1: "칵테일 바", 2: "이탈리아", 3: "키토 식단", 4: "사과", 5: "게임"}
personality_type = get_personality_type(answers)
print(personality_type)


def format_personality_type(personality_type):
    formatted_type = "당신은 {} 유형입니다.".format(", ".join(personality_type))
    return formatted_type


answers = {1: "칵테일 바", 2: "이탈리아", 3: "키토 식단", 4: "사과", 5: "게임"}
personality_type = get_personality_type(answers)
formatted_type = format_personality_type(personality_type)
print(formatted_type)
