import { useState, useEffect } from "react";
import {
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
  ScrollView,
  Modal,
  Platform,
} from "react-native";
import { useForm } from "react-hook-form";

import Icon from "react-native-vector-icons/Ionicons";
// import StarRating from "react-native-star-rating";
import StarRating from "./StarRating";
import { useReactiveVar } from "@apollo/client";
import MIcon from "react-native-vector-icons/MaterialCommunityIcons";
import picData from "./mapped_idx2item.json";

import { emailVar, postApi } from "./Api";

export default function WineInfo({ navigation: { navigate }, route }) {
  const email = useReactiveVar(emailVar);
  const { register, handleSubmit, setValue, watch } = useForm();

  const target = route.params.item;

  const [popupVisible, setPopupVisible] = useState(false);
  const [selectVisible, setSelectVisible] = useState(false);
  const [ratingValue, setRatingValue] = useState(0);
  const [noteVisible, setNoteVisible] = useState(false);
  const [selectedFruit, setSelectedFruit] = useState("Red_Fruit");

  console.log(target);

  const onValid = async (data) => {
    // 존재여부 확인 용도
    const endpoint = "wine/rating_check/";
    try {
      const response = await postApi(endpoint, data);
      console.log(response.data);
      if (response.data == 0) {
        setPopupVisible(true);
      } else {
        setSelectVisible(true);
      }
    } catch (error) {
      alert(error);
      console.log(error);
    }
  };

  const onPost = async (data) => {
    // rating 보내는 용도
    const endpoint = "wine/rating/";
    try {
      const response = await postApi(endpoint, data);
      console.log(response.data);
    } catch (error) {
      alert(error);
      console.log(error);
    }
  };

  useEffect(() => {
    register("email", {
      required: true,
    });
    register("wine_id", {
      required: true,
    });
    register("timestamp", {
      required: true,
    });
    setValue("email", email);
    setValue("wine_id", target.item_id);
    setValue("timestamp", 0);
  }, [register]);

  const onStarRatingPress = (rating) => {
    setRatingValue(rating);
  };

  //console.log(ratingValue);

  const selectAgain = () => {
    return (
      <Modal animationType="slide" transparent={true} visible={selectVisible}>
        <View style={styles.popupWrapper}>
          <View style={styles.popupView}>
            <Text style={{ fontSize: 16 }}>남긴 평점이 이미 존재합니다!</Text>
            <View style={{ flexDirection: "row" }}>
              {ratingPopup()}
              <TouchableOpacity
                onPress={() => {
                  setPopupVisible(true);
                  setSelectVisible(false);
                }}
                style={styles.submitBtn}
              >
                <Text>다시 남기기</Text>
              </TouchableOpacity>
              <TouchableOpacity
                onPress={() => {
                  setSelectVisible(false);
                }}
                style={styles.submitBtn}
              >
                <Text>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    );
  };

  const ratingPopup = () => {
    return (
      <Modal animationType="slide" transparent={true} visible={popupVisible}>
        <View style={styles.popupWrapper}>
          <View style={styles.popupView}>
            <Text style={{ fontSize: 16 }}>와인 평점 남기기</Text>
            <StarRating
              disabled={false}
              maxStars={5}
              rating={ratingValue}
              selectedStar={(rating) => onStarRatingPress(rating)}
              starSize={55}
              halfStarEnabled
              emptyStarColor="#C6C6C6"
              fullStarColor="gold"
              starStyle={{ marginTop: 10, paddingRight: 5 }}
            />
            <View style={{ flexDirection: "row" }}>
              <TouchableOpacity
                onPress={() => {
                  register("rating", {
                    required: true,
                  });
                  setValue("rating", ratingValue);
                  onPost(watch());
                  setPopupVisible(false);
                }}
                style={styles.submitBtn}
              >
                <Text>Submit</Text>
              </TouchableOpacity>
              <TouchableOpacity
                onPress={() => {
                  setPopupVisible(false);
                }}
                style={styles.submitBtn}
              >
                <Text>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    );
  };

  const tasteList = [
    { id: 0, t1: target.Light, name1: "Light", t2: target.Bold, name2: "Bold" },
    {
      id: 1,
      t1: target.Smooth,
      name1: "Smooth",
      t2: target.Tannic,
      name2: "Tannic",
    },
    { id: 2, t1: target.Dry, name1: "Dry", t2: target.Sweet, name2: "Sweet" },
    {
      id: 3,
      t1: target.Soft,
      name1: "Soft",
      t2: target.Acidic,
      name2: "Acidic",
    },
  ];

  const fruitList = {
    Red_Fruit: {
      color: "#FF6464",
      icon: "fruit-cherries",
      notes: ["cherry", "cranberry", "strawberry", "raspberry"],
      name: "Red Fruit",
    },
    Tropical: {
      color: "#FFD053",
      icon: "fruit-pineapple",
      notes: [
        "passion fruit",
        "pineapple",
        "guava",
        "mango",
        "kiwi",
        "lychee",
        "papaya",
        "starfruit",
      ],
      name: "Tropical",
    },
    Tree_Fruit: {
      color: "#AAE074",
      icon: "food-apple",
      notes: [
        "peach",
        "green apple",
        "apple",
        "pear",
        "melon",
        "apricot",
        "nectarine",
      ],
      name: "Tree Fruit",
    },
    Oaky: {
      color: "#AD7153",
      icon: "coffee",
      notes: [
        "butter",
        "vanila",
        "coconut",
        "chocolate",
        "caramel",
        "mocha",
        "tobacco",
      ],
      name: "Oaky",
    },
    Ageing: {
      color: "#D2B8AB",
      icon: "spoon-sugar",
      notes: [
        "almond",
        "peanut",
        "hazelnut",
        "walnut",
        "maple syrup",
        "brown sugar",
      ],
      name: "Ageing",
    },
    Black_Fruit: {
      color: "#73738D",
      icon: "fruit-grapes",
      notes: [
        "blackberry",
        "plum",
        "blueberry",
        "blackcurrant",
        "cassis",
        "olive",
      ],
      name: "Black Fruit",
    },
    Citrus: {
      color: "#FFF48C",
      icon: "fruit-citrus",
      notes: ["grapefruit", "lime", "lemon", "orange", "tangerine"],
      name: "Citrus",
    },
    Dried_Fruit: {
      color: "#CCF7F8",
      icon: "weather-windy",
      notes: ["fig", "raisin", "prune"],
      name: "Dried Fruit",
    },
    Earthy: {
      color: "#635DFF",
      icon: "mushroom",
      notes: ["honey", "stone", "salt", "mushroom", "balsamic"],
      name: "Earthy",
    },
    Floral: {
      color: "#FAA2E8",
      icon: "flower-poppy",
      notes: ["elderflower", "acacia", "jasmine", "lavender", "lilac"],
      name: "Floral",
    },
    Microbio: {
      color: "#F2B26A",
      icon: "bread-slice",
      notes: ["cheese", "cream", "oil", "banana"],
      name: "Microbio",
    },
    Spices: {
      color: "#D80000",
      icon: "chili-hot",
      notes: ["pepper", "mint", "cinamon", "eucalyptus"],
      name: "Spices",
    },
    Vegetal: {
      color: "#3C8833",
      icon: "grass",
      notes: ["gooseberry", "grass", "straw", "asparagus", "tomato"],
      name: "Vegetal",
    },
  };

  const fruitInfo = () => {
    const selectedKeys = [
      "Red_Fruit",
      "Tropical",
      "Tree_Fruit",
      "Oaky",
      "Ageing",
      "Black_Fruit",
      "Citrus",
      "Dried_Fruit",
      "Earthy",
      "Floral",
      "Microbio",
      "Spices",
      "Vegetal",
    ];

    const selectedData = {};
    for (const key of selectedKeys) {
      selectedData[key] = target[key];
    }

    const sortedData = Object.entries(selectedData).sort((a, b) => b[1] - a[1]);
    const top3Keys = sortedData.slice(0, 3).map(([key]) => key); // 가장 높은 3개의 키 추출

    return (
      <View style={styles.fruitContainer}>
        <Modal animationType="none" transparent={true} visible={noteVisible}>
          <View style={styles.popupWrapper}>
            <View style={styles.popupView}>
              <View style={{ alignItems: "center" }}>
                <Text
                  style={{ fontSize: 40, fontWeight: "700", marginBottom: 20 }}
                >
                  {fruitList[selectedFruit].name}
                </Text>
                <Text style={{ marginBottom: 20 }}>Taste Like...</Text>
                {fruitList[selectedFruit].notes.map((note, index) => {
                  return (
                    <Text key={index} style={{ fontSize: 20 }}>
                      {note}
                    </Text>
                  );
                })}
              </View>
              <TouchableOpacity
                onPress={() => {
                  setNoteVisible(false);
                }}
                style={styles.submitBtn}
              >
                <Text>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
        {top3Keys.map((fruit, key) => (
          <TouchableOpacity
            key={key}
            style={[
              styles.iconRound,
              { backgroundColor: fruitList[fruit].color },
            ]}
            onPress={() => {
              setSelectedFruit(fruit);
              setNoteVisible(true);
            }}
          >
            <MIcon
              name={fruitList[fruit].icon}
              size={50}
              color={"#FFFFFF"}
            ></MIcon>
            <Text style={{ fontSize: 10 }}>{fruitList[fruit].name}</Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const tasteInfo = () => {
    return (
      <View style={styles.tempBox}>
        {tasteList.map((taste, index) => {
          if (taste.t1 + taste.t2 > 90) {
            return (
              <View
                key={index}
                style={{
                  flexDirection: "row",
                  marginTop: 10,
                  marginBottom: 10,
                  alignItems: "center",
                }}
              >
                <View style={{ width: "15%" }}>
                  <Text style={{ textAlign: "right" }}>{taste.name1}</Text>
                </View>
                <View style={styles.barOut}>
                  <View style={barStyle(`${taste.t2 - 5}%`).barIn}></View>
                </View>
                <View style={{ width: "15%" }}>
                  <Text>{taste.name2}</Text>
                </View>
              </View>
            );
          }
        })}
        {/*
        <Text style={{ marginTop: 10 }}>{target.pairing}</Text>
      */}
      </View>
    );
  };

  return (
    <View style={{ flex: 1 }}>
      <ScrollView style={styles.wrapper}>
        <Image
          style={styles.image}
          source={
            picData[target.item_id]
              ? { uri: picData[target.item_id] }
              : require("./assets/wine.jpg")
          }
        />
        <Text style={styles.name}>{target.name}</Text>
        <View style={styles.ratingANDprice}>
          <View style={{ flexDirection: "row" }}>
            <Icon name={"star"} size={20} color={"#000000"} />
            <Text style={{ fontSize: 18 }}>
              {" "}
              {target.wine_rating} ({target.num_votes})
            </Text>
          </View>
          <Text style={{ fontSize: 18 }}>가격: $ {target.price}</Text>
        </View>
        <View style={styles.country}>
          <Text style={{ fontSize: 18, fontWeight: "700" }}>From </Text>
          <Text style={{ fontSize: 15 }}>
            {" "}
            {target.country}, {target.region1}
          </Text>
        </View>
        <View style={{ margin: 5, marginBottom: 10 }}>
          <Text style={styles.detail}>와이너리: {target.winery}</Text>
          <Text style={styles.detail}>포도: {/*target.grape*/}</Text>
          <Text style={styles.detail}>빈티지: {target.vintage}</Text>
          <Text style={styles.detail}>와인 타입: {target.winetype}</Text>
        </View>
        {tasteInfo()}
        {fruitInfo()}
        {ratingPopup()}
        {selectAgain()}
        <TouchableOpacity
          onPress={() => {
            onValid(watch());
          }}
          style={styles.ratingBtn}
        >
          <Text>평점 남기기</Text>
        </TouchableOpacity>

        <View style={{ height: 50 }} />
      </ScrollView>
    </View>
  );
}

const barStyle = (completed) =>
  StyleSheet.create({
    barIn: {
      height: "100%",
      backgroundColor: "#F48FB1",
      borderRadius: 100,
      width: "10%",
      marginLeft: completed,
    },
  });

const styles = StyleSheet.create({
  wrapper: {
    marginLeft: "5%",
    marginRight: "5%",
  },
  image: {
    width: "100%",
    height: 350,
    resizeMode: "contain",
    //borderColor: "#D9D9D9",
    //borderWidth: 2,
    marginTop: 10,
    marginBottom: 15,
  },
  name: {
    fontSize: 35,
    alignItems: "flex-start",
    margin: 5,
  },
  ratingANDprice: {
    flexDirection: "row",
    margin: 5,
    alignItems: "center",
    justifyContent: "space-between",
  },
  country: {
    flexDirection: "row",
    margin: 5,
    alignItems: "center",
  },
  detail: {
    fontSize: 15,
    color: "#707070",
  },
  text: {
    fontSize: 20,
    margin: 5,
  },
  tempBox: {
    //margin: 5,
    paddingTop: 5,
    paddingBottom: 15,
    //height: "10%",
    //backgroundColor: "#D9D9D9",
    // alignItems: "center",
    justifyContent: "center",
  },
  popupWrapper: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 22,
  },
  popupView: {
    margin: 20,
    backgroundColor: "white",
    borderRadius: 20,
    padding: 35,
    alignItems: "center",
    shadowColor: "#000",
    //그림자의 영역 지정
    shadowOffset: {
      width: 0,
      height: 2,
    },
    //불투명도 지정
    shadowOpacity: 0.25,
    //반경 지정
    shadowRadius: 3.84,
  },
  popupText: {
    marginBottom: 15,
    textAlign: "center",
  },
  ratingBtn: {
    width: "50%",
    borderRadius: 25,
    height: 50,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F48FB1",
    marginLeft: "25%",
    marginRight: "25%",
    marginTop: 5,
  },
  submitBtn: {
    width: 90,
    borderRadius: 25,
    height: 40,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F48FB1",
    marginRight: 15,
    marginLeft: 15,
    marginTop: 30,
  },
  barOut: {
    height: 20,
    backgroundColor: "#DAF7A6",
    borderRadius: 100,
    marginRight: 10,
    marginLeft: 10,
    ...Platform.select({
      ios: {
        width: "60%",
      },
      android: {
        width: "60%",
      },
      web: {
        width: 300,
      },
    }),
  },
  fruitContainer: {
    paddingTop: 5,
    paddingBottom: 10,
    paddingRight: 20,
    justifyContent: "space-around",
    flexDirection: "row",
    margin: 10,
    marginBottom: 20,
  },
  iconRound: {
    width: 90,
    height: 90,
    borderRadius: 45,
    alignItems: "center",
    justifyContent: "center",
  },
});
