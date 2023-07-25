import { useState, useEffect } from "react";
import {
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
  ScrollView,
  Modal,
} from "react-native";
import { useForm } from "react-hook-form";

import Icon from 'react-native-vector-icons/Ionicons';
// import StarRating from "react-native-star-rating";
import { useReactiveVar } from "@apollo/client";

import { emailVar, postApi } from "./Api";

export default function WineInfo ({navigation: {navigate}, route}) {
  const email = useReactiveVar(emailVar);
  const { register, handleSubmit, setValue, watch } = useForm();

  const target = route.params.item;
  const [popupVisible, setPopupVisible] = useState(false);
  const [selectVisible, setSelectVisible] = useState(false);
  const [ratingValue, setRatingValue] = useState(0);
  
  console.log(target)

  const onValid = async (data) => { // 존재여부 확인 용도
		const endpoint = "wine/rating_check/";
		try {
			const response = await postApi(endpoint, data);
      console.log(response.data)
      if (response.data == 0) {
        setPopupVisible(true);
      }
      else {
        setSelectVisible(true);
      }
		} catch (error) {
			alert(error);
			console.log(error);
		}
	};

  const onPost = async (data) => { // rating 보내는 용도
		const endpoint = "wine/rating/";
		try {
			const response = await postApi(endpoint, data);
      console.log(response.data)
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
    setValue("wine_id", target.item_id)
    setValue("timestamp", 0)
	}, [register]);

  const onStarRatingPress = (rating) => {
    setRatingValue(rating);
  };

  //console.log(ratingValue);

  const selectAgain = () => {
    return (
      <Modal
        animationType="slide"
        transparent={true}
        visible={selectVisible}
      >
        <View style={styles.popupWrapper}>
          <View style={styles.popupView}>
            <Text style={{fontSize: 16}}>남긴 평점이 이미 존재합니다!</Text>
            <View style={{flexDirection:'row'}}>
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
    )
  }

  
  const ratingPopup = () => {

    return (
      <Modal
        animationType="slide"
        transparent={true}
        visible={popupVisible}
      >
        <View style={styles.popupWrapper}>
          <View style={styles.popupView}>
            <Text style={{fontSize: 16}}>와인 평점 남기기</Text>
            {/* <StarRating
              disabled={false}
              maxStars={5}
              rating={ratingValue}
              selectedStar={(rating) => onStarRatingPress(rating)}
              starSize={50}
              halfStarEnabled
              emptyStarColor="#C6C6C6"
              fullStarColor="gold"
              starStyle={{marginTop: 10, paddingRight: 5}}
            /> */}
            <View style={{flexDirection:'row'}}>
              <TouchableOpacity
                onPress={() => {
                  register("rating", {
                    required: true,
                  });
                  setValue("rating", ratingValue)
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
    )
  }

  const tasteList =[
    {id: 0, t1: target.Light, name1: "Light", t2: target.Bold, name2: "Bold"}, 
    {id: 1, t1: target.Smooth, name1: "Smooth", t2: target.Tannic, name2: "Tannic"}, 
    {id: 2, t1: target.Dry, name1: "Dry", t2: target.Sweet, name2: "Sweet"}, 
    {id: 3, t1: target.Soft, name1: "Soft", t2: target.Acidic, name2: "Acidic"}
  ]

  const tasteInfo = () => {
    return (
      <View style={styles.tempBox}>
        {tasteList.map((taste, index) => {
          if ((taste.t1 + taste.t2) > 90) {
            return (
              <View key={index} style={{flexDirection: 'row', marginTop: 10, marginBottom: 10, alignItems: "center"}}>
                <View style={{marginLeft: 15, width:"10%"}}>
                  <Text>{taste.name1}</Text>
                </View>
                <View style={styles.barOut}>
                  <View style={barStyle(`${(taste.t2)-5}%`).barIn}>
                  </View>
                </View>
                <View style={{width:"15%"}}>
                  <Text>{taste.name2}</Text>
                </View>
              </View>
            )
          }
        })}
        <Text style={{marginTop: 10}}>{target.pairing}</Text>
      </View>
    )
  }

  return (
    <ScrollView style={styles.wrapper}>
      <Image style={styles.image} source={require('./assets/고라파덕.jpg')} />
      <Text style={styles.name}>{target.name}</Text>
      <View style={styles.ratingANDprice}>
        <View style={{flexDirection: "row"}}>
          <Icon name={'star'} size={20} color={"#000000"}/>
          <Text style={{fontSize: 18}}>  {target.wine_rating} ({target.num_votes})</Text>
        </View>
        <Text style={{fontSize: 18}}>가격: $ {target.price}</Text>
      </View>
      <View style={styles.country}>
        <Text style={{fontSize: 18, fontWeight: "700"}}>From   </Text>
        <Text style={{fontSize: 15}}>  {target.country},  {target.region1}</Text>
      </View>
      <View style={{margin:5, marginBottom:10}}>
        <Text style={styles.detail}>와이너리:  {target.winery}</Text>
        <Text style={styles.detail}>포도:  {/*target.grape*/}</Text>
        <Text style={styles.detail}>빈티지:  {target.vintage}</Text>
        <Text style={styles.detail}>와인 타입:  {target.winetype}</Text>
      </View>
      <Text style={styles.text}>{/*target.text*/}와인 설명입니다.</Text>
      {tasteInfo()}
      {ratingPopup()}
      {selectAgain()}
      <TouchableOpacity
        onPress={() => {onValid(watch())}}
        style={styles.ratingBtn}
      >
        <Text>평점 남기기</Text>
      </TouchableOpacity>

      
      <View style={{height: 80}}/>
    </ScrollView>
  );
}

const barStyle = (completed) => StyleSheet.create({
  barIn: {
    height: "100%",
    backgroundColor: "#F48FB1",
    borderRadius: 100,
    width : "10%",
    marginLeft: completed
  },
})

const styles = StyleSheet.create({
  wrapper: {
    //flex: 1,
    marginLeft: "5%",
    marginRight: "5%",
  },
  image: {
    width: "100%",
    height: 350,
    resizeMode: "contain",
    borderColor: "#D9D9D9",
    borderWidth: 2,
    marginTop: 10,
    marginBottom: 15,
  },
  name: {
    fontSize: 35,
    alignItems: "flex-start",
    margin: 5
  },
  ratingANDprice: {
    flexDirection: "row",
    margin: 5,
    alignItems: "center",
    justifyContent: "space-between"
  },
  country: {
    flexDirection: "row",
    margin: 5,
    alignItems: "center"
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
    margin: 5,
    paddingTop: 5,
    paddingBottom: 15,
    //height: "10%",
    //backgroundColor: "#D9D9D9",
    alignItems: "center",
    justifyContent: "center"
  },
  popupWrapper: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop:22,
  },
  popupView: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 20,
    padding: 35,
    alignItems: 'center',
    shadowColor: '#000',
    //그림자의 영역 지정
    shadowOffset: {
      width: 0,
      height:2
    },
    //불투명도 지정
    shadowOpacity: 0.25,
    //반경 지정
    shadowRadius: 3.84,
  },
  popupText: {
    marginBottom: 15,
    textAlign: 'center',
  },
  ratingBtn: {
    width: "50%",
    borderRadius: 25,
    height: 50,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#F48FB1",
    marginLeft:"25%",
    marginRight:"25%",
    marginTop: 5
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
    marginTop: 30
  },
  barOut: {
    height: 20,
    width: "60%",
    backgroundColor: "#DAF7A6",
    borderRadius: 100,
    marginRight: 10,
    marginLeft: 10
  },
})