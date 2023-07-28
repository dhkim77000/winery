import React from "react";
import { StyleSheet, Text, View, Image, TouchableOpacity } from "react-native";
import Swiper from "react-native-web-swiper";
import { logUserOut } from "./Api";

export default function Home() {
  return (
    <View style={styles.container}>
      <View style={styles.banner}>{swipers()}</View>
      <Text style={styles.paragraph}>WINERY에 오신 걸 환영합니다</Text>
      <Text style={styles.paragraph}>아래 하트를 눌러</Text>
      <Text style={styles.paragraph}>개인화된 와인 추천을 받아보세요</Text>
      <TouchableOpacity style={styles.tempbtn} onPress={() => logUserOut()}>
        <Text>Log Out</Text>
      </TouchableOpacity>
    </View>
  );
}

export function swipers() {
  // 배너
  return (
    <Swiper
      loop={true}
      timeout={4}
      controlsProps={{ prevPos: false, nextPos: false }}
    >
      <View style={styles.slide}>
        <Image
          style={styles.image}
          source={require("./assets/banner/banner2.png")}
        />
      </View>
      <View style={styles.slide}>
        <Image
          style={styles.image}
          source={require("./assets/banner/banner4.png")}
        />
      </View>
      <View style={styles.slide}>
        <Image
          style={styles.image}
          source={require("./assets/banner/banner7.jpg")}
        />
      </View>
      <View style={styles.slide}>
        <Image
          style={styles.image}
          source={require("./assets/banner/banner6.jpg")}
        />
      </View>
      <View style={styles.slide}>
        <Image
          style={styles.image}
          source={require("./assets/banner/banner5.png")}
        />
      </View>
      <View style={styles.slide}>
        <Image
          style={styles.image}
          source={require("./assets/banner/banner1.png")}
        />
      </View>
    </Swiper>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
  },
  paragraph: {
    marginBottom: 10,
    fontSize: 25,
    fontWeight: "300",
    alignItems: "flex-start",
  },
  image: {
    flex: 1,
    resizeMode: "cover",
    width: "100%",
    height: "100%",
  },
  slide: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  tempbtn: {
    width: "80%",
    borderRadius: 25,
    height: 50,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 20,
    marginBottom: 20,
    backgroundColor: "#ffc0cb",
  },
  banner: {
    height: "50%",
    width: "110%",
    //resizeMode: "cover",
    justifyContent: "center",
    //backgroundColor: "#FFFFFF",
    padding: 10,
    marginBottom: 20,
  },
});
