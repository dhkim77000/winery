const questions = [
  {
    unique_id: 0,
    question: "어떤 스타일의 와인을 가장 드시고 싶나요?",
    answers: [
      {
        text: "다양한 베리류, 오크, 가죽 향과 복합적인 풍미가 느껴지는 맛",
        postLabel: "a1",
        label: 1,
        styles: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
      },
      {
        text: "열대 과일, 시트러스 향과 청량감과 상큼함이\n느껴지는 맛",
        postLabel: "a2",
        label: 2,
        styles: [4, 5, 6, 7, 8, 9],
      },
      {
        text: "말린 과일, 탄산감과 청량함이 느껴지는 맛",
        postLabel: "a3",
        label: 3,
        styles: [0, 1, 2, 3],
      },
    ],
  },
  {
    unique_id: 1,
    question: "와인을 마시고 원하는 입 안의 상태는 ?",
    answers: [
      {
        text: "풍부하고\n깊은\n입안 감촉",
        postLabel: "b1",
        label: 1,
        styles: [0, 4, 11, 21],
      },
      {
        text: "\n\n",
        postLabel: "b2",
        label: 2,
        styles: [1, 2, 6, 10, 12, 13, 14, 15, 18, 20, 22],
      },
      { text: "\n\n", postLabel: "b3", label: 3, styles: [5, 7, 8, 9, 16, 19] },
      {
        text: "상쾌하고\n가벼운\n입안 감촉",
        postLabel: "b4",
        label: 4,
        styles: [3, 17],
      },
    ],
  },
  {
    unique_id: 2,
    question: "와인을 마실 때, 생각하는 가격대는?",
    answers: [
      { text: "5만원 이하", postLabel: "c1", label: 1, styles: [2, 3, 16, 17] },
      {
        text: "5만원 ~ 15만원",
        postLabel: "c2",
        label: 2,
        styles: [9, 20, 21, 22],
      },
      {
        text: "15만원 ~ 30만원",
        postLabel: "c3",
        label: 3,
        styles: [4, 5, 8, 10, 11, 12],
      },
      {
        text: "30만원 이상",
        postLabel: "c4",
        label: 4,
        styles: [0, 1, 6, 7, 13, 14, 15, 18],
      },
    ],
  },
  {
    unique_id: 3,
    question: "와인을 마실 때, 선호하는 맛은?",
    answers: [
      {
        text: "묵직하고 힘이 느껴지는 맛 ",
        postLabel: "d1",
        label: 1,
        styles: [0, 13, 15, 18, 20, 21, 22],
      },
      {
        text: "상쾌하고 활기찬 맛",
        postLabel: "d2",
        label: 2,
        styles: [2, 3, 5, 8, 9, 16, 17],
      },
      {
        text: "세련되고 우아한 맛",
        postLabel: "d3",
        label: 3,
        styles: [1, 6, 7, 19],
      },
      {
        text: "다채롭고 섬세한 맛",
        postLabel: "d4",
        label: 4,
        styles: [4, 10, 11, 12, 14],
      },
    ],
  },
  {
    unique_id: 4,
    question: "와인을 마실 때, 어울릴 것 같은 음식은?",
    answers: [
      {
        text: "해산물",
        postLabel: "e1",
        label: 1,
        styles: [4, 5, 6, 7, 8, 9],
      },
      {
        text: "고기",
        postLabel: "e2",
        label: 2,
        styles: [10, 11, 12, 15, 20, 21, 22],
      },
      {
        text: "다양한 음식이 포함된 파인 다이닝",
        postLabel: "e3",
        label: 3,
        styles: [13, 14, 16, 17, 18, 19],
      },
      { text: "디저트 치즈", postLabel: "e4", label: 4, styles: [0, 1, 2, 3] },
    ],
  },
  {
    unique_id: 5,
    question: "당신이 선호하는 선물 스타일은?",
    answers: [
      { text: "감동적인 선물", label: 1, styles: [1, 6, 8, 15, 18, 22] },
      {
        text: "유용한 선물",
        label: 2,
        styles: [2, 4, 5, 9, 10, 11, 12, 16, 17, 21],
      },
      { text: "독특한 선물", label: 3, styles: [0, 3] },
      { text: "세련된 선물", label: 4, styles: [7, 13, 14, 19, 20] },
    ],
  },
  {
    unique_id: 6,
    question:
      "오랜만에 친구들과 모이는 날이 왔습니다.\n 어떤 모임을 원하시나요?",
    answers: [
      {
        text: "가볍게 만나서 화목하게 떠들면서 편안하게 보내고 싶어요",
        label: 1,
        styles: [4, 5, 9, 16, 17],
      },
      {
        text: "친구들과 함께 활동적인 액티비티를 즐기고 싶어요",
        label: 2,
        styles: [11, 20, 21, 22],
      },
      {
        text: "조용한 레스토랑에서 오붓하게 시간을 보내고 싶어요",
        label: 3,
        styles: [6, 7, 10, 12, 13, 14, 15, 18, 19],
      },
      {
        text: "신나는 파티를 열어서 같이 즐기고 싶어요",
        label: 4,
        styles: [0, 1, 2, 3, 8],
      },
    ],
  },
  {
    unique_id: 7,
    question: "여행 중에 어떤 경험을 즐기고 싶으신가요?",
    answers: [
      {
        text: "숨겨진 보석같은 장소를 찾아서 현지인과 함께 놀며 즐기고 싶어요",
        label: 1,
        styles: [4, 8, 9, 10, 16, 17],
      },
      {
        text: "아름다운 풍경을 감상하며 고기를 구우면서 캠핑을 하고 싶어요",
        label: 2,
        styles: [11, 20, 21, 22],
      },
      {
        text: "멋진 드레스와 수트를 입고 파티에 참여하고 싶어요",
        label: 3,
        styles: [0, 1, 2, 3, 5],
      },
      {
        text: "역사와 전통이 묻어나는 곳을 방문하여 오랜 세월의 흔적을 느끼고 싶어요",
        label: 4,
        styles: [6, 7, 12, 13, 14, 15, 18, 19],
      },
    ],
  },
  {
    unique_id: 8,
    question: "휴일이나 주말, 나의 취향은?",
    answers: [
      {
        text: "아늑한 공간에서 혼자 책을 읽거나 음악을 듣는다",
        label: 1,
        styles: [6, 7, 10, 12, 14, 18, 19],
      },
      {
        text: "친구들과 함께 액티비티를 즐긴다",
        label: 2,
        styles: [0, 2, 3, 9, 11, 16, 17, 20, 21],
      },
      { text: "유명한 맛집을 찾아다닌다", label: 3, styles: [1, 4, 5, 8] },
      {
        text: "전시회나 문화적인 장소에 참여한다",
        label: 4,
        styles: [13, 15, 22],
      },
    ],
  },
  {
    unique_id: 9,
    question: "좋아하는 옷 스타일은?",
    answers: [
      {
        text: "클래식하고 우아한 스타일",
        label: 1,
        styles: [1, 6, 13, 14, 15, 18, 19],
      },
      { text: "밝고 화사한 스타일", label: 2, styles: [2, 3, 7, 9, 16, 17] },
      { text: "남성적이고 마초적인 스타일", label: 3, styles: [20, 21, 22] },
      {
        text: "트렌디하고 개성있는 스타일",
        label: 4,
        styles: [0, 4, 5, 8, 10, 11, 12],
      },
    ],
  },
];

export default questions;
