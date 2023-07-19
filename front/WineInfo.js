import { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
} from "react-native";

import wine from "./wineList";
import Icon from 'react-native-vector-icons/Ionicons';

export default function WineInfo ({navigation: {navigate}, route}) {
  const id = route.params.id
  const target = wine[id-1]
  console.log(id) // 확인용
  return (
    <View style={styles.wrapper}>
      <Image style={styles.image} source={target.pic} />
      <Text style={styles.name}>{target.name}</Text>
      <View style={styles.ratingANDprice}>
        <View style={{flexDirection: "row"}}>
          <Icon name={'star'} size={20} color={"#000000"}/>
          <Text style={{fontSize: 18}}>  {target.rating} ({target.numVote})</Text>
        </View>
        <Text style={{fontSize: 18}}>가격: {target.price}</Text>
      </View>
      <View style={styles.country}>
        <Text style={{fontSize: 18, fontWeight: "700"}}>From   </Text>
        <Text style={{fontSize: 15}}>  {target.country},  {target.region}</Text>
      </View>
      <View style={{margin:5, marginBottom:10}}>
        <Text style={styles.detail}>와이너리:  {target.winery}</Text>
        <Text style={styles.detail}>포도:  {target.grape}</Text>
        <Text style={styles.detail}>빈티지:  {target.vintage}</Text>
        <Text style={styles.detail}>와인 타입:  {target.wineType}</Text>
      </View>
      <Text style={styles.text}>{target.text}</Text>
      <View style={styles.tempBox}>
        <Text>Taste, Pairing 정보</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    flex: 1,
    marginLeft: "5%",
    marginRight: "5%"
  },
  image: {
    width: "100%",
    height: "42%",
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
    fontSize: 15,
    margin: 5,
  },
  tempBox: {
    margin: 5,
    height: "10%",
    backgroundColor: "#D9D9D9",
    alignItems: "center",
    justifyContent: "center"
  }
})