import React from "react";
import { StyleSheet, Text, View, Image, TouchableOpacity } from "react-native";
import Swiper from "react-native-web-swiper";
import { logUserOut } from "./Api";

export default function Home() {
	return (
		<View style={styles.container}>
			<View style={styles.banner}>{swipers()}</View>
			<Text style={styles.paragraph}>금주의 추천 와인</Text>
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
				<Image style={styles.image} source={require("./assets/고라파덕.jpg")} />
			</View>
			<View style={styles.slide}>
				<Image style={styles.image} source={require("./assets/꼬부기.jpg")} />
			</View>
			<View style={styles.slide}>
				<Image style={styles.image} source={require("./assets/이브이.jpg")} />
			</View>
			<View style={styles.slide}>
				<Image style={styles.image} source={require("./assets/케이시.jpg")} />
			</View>
			<View style={styles.slide}>
				<Image style={styles.image} source={require("./assets/파이리.jpg")} />
			</View>
		</Swiper>
	);
}

const styles = StyleSheet.create({
	wrapper: {
		justifyContent: "center",
	},
	container: {
		flex: 1,
		alignItems: "center",
	},
	paragraph: {
		margin: 24,
		fontSize: 25,
		fontWeight: "bold",
		alignItems: "flex-start",
	},
	image: {
		flex: 1,
		resizeMode: "contain",
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
		width: "80%",
		resizeMode: "contain",
		justifyContent: "center",
		backgroundColor: "#FFFFFF",
		padding: 10,
	},
});
