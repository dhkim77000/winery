import React, { useEffect, useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Platform,
  Image,
  ScrollView,
} from "react-native";
import questions from "./MbtiQ";
import mbtiStyle from "./MbtiS";
import { useForm } from "react-hook-form";
import { postApi } from "./Api";
import Icon from "react-native-vector-icons/MaterialCommunityIcons";

export default function Mbti({ navigation: { navigate, goBack }, route }) {
  //console.log(route.params.email);
  //console.log(route.params.password);
  const { register, setValue, watch } = useForm();
  const [questionNum, setQuestionNum] = useState(0); // 문항번호 0~9
  const [result, setResult] = useState([]); // 고른 결과 리스트
  const [count, setCount] = useState(Array.from({ length: 23 }, () => 0));
  const [select, setSelect] = useState(""); // 사진 성별 선택
  const [ran, setRan] = useState(0);
  const [postResult, setPostResult] = useState([]); // 보내줄 결과 리스트
  const [finalId, setFinalId] = useState(null);

  useEffect(() => {
    register("email", {
      required: true,
    });
    register("password", {
      required: true,
    });
    register("mbti_result", {
      required: true,
    });
    register("wine_style", {
      required: true,
    });
    setValue("email", route.params.email);
    setValue("password", route.params.password);
    setRan(Math.random()); // 랜덤 상수
  }, []);

  const isPress = (item) => {
    // 선택지 눌렀을 때
    item.styles.map((id) => {
      count[id]++;
      setCount([...count]);
    });
    if (questionNum < 5) {
      setPostResult([...postResult, item.postLabel]);
    }
    if (questionNum === 9) {
      setValue("mbti_result", postResult);
      const indexResult = []; // maxValue의 인덱스값이 여러개일 상황고려하여 배열에 저장
      const maxValue = Math.max(...count);
      let maxIndex = count.indexOf(maxValue);
      while (maxIndex != -1) {
        indexResult.push(maxIndex);
        maxIndex = count.indexOf(maxValue, maxIndex + 1);
      }
      setFinalId(indexResult[Math.floor(ran * indexResult.length)]);
    }
    setResult([...result, item.label]);
    setQuestionNum(questionNum + 1);
  };

  const onValid = async (data) => {
    const endpoint = "login/register/";
    console.log(data);
    try {
      const response = await postApi(endpoint, data);
      //console.log(response.data);
    } catch (error) {
      alert(error);
      console.log(error);
    }
  };

  const isBack = () => {
    // back 눌렀을 때 전 문제로 돌아가기
    if (questionNum != 0) {
      postResult.pop();
      setPostResult([...postResult]);
      const back = result.pop();
      if (questions[questionNum - 1].answers[back - 1].styles) {
        questions[questionNum - 1].answers[back - 1].styles.map((id) => {
          count[id]--;
          setCount([...count]);
        });
      } else {
        console.warn("Styles not found for the selected answer.");
      }
      setResult([...result]);
      setQuestionNum(questionNum - 1);
    } else {
      goBack();
    }
  };

  //console.log(result); // 결과 잘 나오는지 실험용
  //console.log(postResult);

  if (questionNum === 11) {
    //완료 페이지

    return (
      <View style={{ flex: 1 }}>
        <ScrollView
          style={{
            backgroundColor: "#FFC0CB",
          }}
          contentContainerStyle={{
            alignItems: "center",
            //justifyContent: "center",
          }}
        >
          <View
            style={{
              flexDirection: "row",
              justifyContent: "center",
              margin: 10,
              marginTop: 30,
            }}
          >
            <TouchableOpacity
              onPress={() => {
                setSelect(mbtiStyle[finalId].mPic);
              }}
              style={{ marginRight: 20, alignItems: "center" }}
            >
              <Text style={{ fontSize: 10 }}>man</Text>
              <Icon name={"face-man"} size={50} color={"#000000"} />
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => {
                setSelect(mbtiStyle[finalId].wPic);
              }}
              style={{ marginLeft: 20, alignItems: "center" }}
            >
              <Text style={{ fontSize: 10 }}>woman</Text>
              <Icon name={"face-woman"} size={50} color={"#000000"} />
            </TouchableOpacity>
          </View>
          <Image
            style={{
              marginTop: 10,
              marginBottom: 20,
              ...Platform.select({
                ios: { width: 300, height: 300 },
                android: { width: 300, height: 300 },
                web: { width: 450, height: 450 },
              }),
            }}
            source={select === "" ? setSelect(mbtiStyle[finalId].mPic) : select}
          />
          <Text>당신을 닮은 와인스타일은</Text>
          <Text
            style={{
              fontSize: 30,
              margin: 20,
              textAlign: "center",
            }}
          >
            {mbtiStyle[finalId].mStyle}
          </Text>
          <Text style={{ width: "75%", fontSize: 15, lineHeight: 28 }}>
            {mbtiStyle[finalId].text}
          </Text>
          <TouchableOpacity
            style={styles.doneBtn}
            onPress={() => {
              navigate("Login");
            }}
          >
            <Text>처음으로 돌아가기</Text>
          </TouchableOpacity>
          <View style={{ height: 80 }} />
        </ScrollView>
      </View>
    );
  } else if (questionNum === 10) {
    return (
      <View
        style={[
          styles.wrapper,
          { justifyContent: "center", alignItems: "center" },
        ]}
      >
        <TouchableOpacity style={styles.lastBtn} onPress={() => isBack()}>
          <Text style={{ fontSize: 20 }}>back</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.lastBtn}
          onPress={() => {
            if (finalId) {
              setValue("wine_style", mbtiStyle[finalId].mStyle);
              onValid(watch());
              setQuestionNum(questionNum + 1);
            }
          }}
        >
          <Text style={{ fontSize: 20 }}>결과</Text>
          <Text style={{ fontSize: 20 }}>보러가기</Text>
        </TouchableOpacity>
      </View>
    );
  } else if (questionNum === 1) {
    return (
      <View style={styles.wrapper}>
        <View style={styles.progressBarOut}>
          <View
            style={barStyle(`${(questionNum + 1) * 10}%`).progressBarIn}
          ></View>
        </View>
        <Text style={styles.title}>{questions[questionNum].question}</Text>
        <View
          style={[
            styles.subContainer,
            {
              flexDirection: "row",
              justifyContent: "center",
              textAlign: "center",
              paddingBottom: 20,
            },
          ]}
        >
          {questions[questionNum].answers &&
            questions[questionNum].answers.map(
              // map undefined 오류 방지
              (item) => {
                return (
                  <View key={item.label}>
                    <TouchableOpacity
                      onPress={() => isPress(item)}
                      style={{
                        ...Platform.select({
                          web: {
                            width: 60,
                            height: 60,
                            borderRadius: 30,
                            margin: 18,
                          },
                          ios: {
                            width: 40,
                            height: 40,
                            borderRadius: 20,
                            margin: 15,
                          },
                          android: {
                            width: 55,
                            height: 55,
                            borderRadius: 27.5,
                            margin: 10,
                          },
                        }),
                        borderColor: "#FFC0CB",
                        borderWidth: 2,
                        alignItems: "center",
                        justifyContent: "center",
                        marginBottom: 30,
                        marginTop: 30,
                      }}
                      activeOpacity={0}
                    ></TouchableOpacity>
                    <Text style={{ textAlign: "center" }}>{item.text}</Text>
                  </View>
                );
              }
            )}
        </View>
        <View style={styles.backContainer}>
          <TouchableOpacity style={styles.backBtn} onPress={() => isBack()}>
            <Text style={{ fontSize: 20 }}>back</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  } else if ((questionNum < 10) & (questionNum > -1)) {
    return (
      <View style={styles.wrapper}>
        <View style={{ height: 50 }} />
        <View style={styles.progressBarOut}>
          <View
            style={barStyle(`${(questionNum + 1) * 10}%`).progressBarIn}
          ></View>
        </View>
        <Text style={styles.title}>{questions[questionNum].question}</Text>
        <View style={styles.subContainer}>
          {questions[questionNum].answers.map((item) => {
            return (
              <TouchableOpacity
                key={item.label}
                onPress={() => isPress(item)}
                style={{
                  width: "80%",
                  padding: 20,
                  margin: 15,
                  borderColor: "#FFC0CB",
                  borderWidth: 2,
                  borderRadius: 20,
                  // alignItems: "center", // text 가운데 정렬
                }}
                activeOpacity={0}
              >
                <Text>{item.text}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
        <View style={styles.backContainer}>
          <TouchableOpacity style={styles.backBtn} onPress={() => isBack()}>
            <Text style={{ fontSize: 20 }}>back</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }
}

const barStyle = (completed) =>
  StyleSheet.create({
    progressBarIn: {
      height: "100%",
      backgroundColor: "#F48FB1",
      borderRadius: 100,
      width: completed,
    },
  });

const styles = StyleSheet.create({
  wrapper: {
    flex: 1,
    backgroundColor: "#FFC0CB",
    alignItems: "center",
    justifyContent: "center",
  },
  backContainer: {
    alignItems: "flex-start",
    justifyContent: "center",
    width: "70%",
  },
  title: {
    textAlign: "center",
    marginBottom: 10,
    marginTop: 30,
    fontSize: 20,
  },
  subContainer: {
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    width: "80%",
    margin: 20,
    alignItems: "center",
  },
  backBtn: {
    width: 80,
    borderRadius: 40,
    height: 80,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F48FB1",
    marginTop: 30,
  },
  lastBtn: {
    width: "25%",
    borderRadius: 30,
    height: 80,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F48FB1",
    margin: 30,
    padding: 10,
  },
  doneBtn: {
    width: "80%",
    borderRadius: 25,
    height: 50,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 40,
    backgroundColor: "#F48FB1",
  },
  progressBarOut: {
    height: 20,
    width: "80%",
    backgroundColor: "#FFFFFF",
    borderRadius: 100,
    //marginTop: 50,
  },
});
