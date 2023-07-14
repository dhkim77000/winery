import { StatusBar } from "expo-status-bar";
import React, { useState, useEffect, useRef } from "react";
import { useForm } from "react-hook-form";
import {
	StyleSheet,
	Text,
	View,
	Image,
	TextInput,
	Button,
	TouchableOpacity,
} from "react-native";
import { postApi, getApi } from "./Api";

export default function Login({ navigation }) {
	const { register, handleSubmit, setValue, watch } = useForm();
	const passwordRef = useRef();
	const [isLogIn, setIsLogIn] = useState(false);

	const onNext = (nextOne) => {
		nextOne?.current?.focus();
	};
	const onValid = async (data) => {
		const regex = /\w+@\w+\.[\w,\.]+/;
		if (regex.test(data.email)) {
			await postApi("temp/login/", data)
				.then((response) => {
					console.log(response.data.status)
					if (response.data.status) {
						setIsLogIn(true);
					}
				})
				.catch((error) => {
					alert(error.status);
				});

			if (isLogIn) {
				navigation.navigate("Recommend");
			} else {
				alert("아이디/이메일을 찾을 수 없습니다");
			}
		} else {
			alert("이메일 형식이 맞지 않습니다");
		}
	};

	useEffect(() => {
		register("email", {
			required: true,
		});
		register("password", {
			required: true,
		});
	}, [register]);

	return (
		<View style={styles.container}>
			<Image style={styles.image} source={require("./assets/logo.png")} />
			<StatusBar style="auto" />
			<View style={styles.inputView}>
				<TextInput
					style={styles.TextInput}
					placeholder="Email"
					placeholderTextColor="#003f5c"
					autoCapitalize={"none"}
					onChangeText={(text) => setValue("email", text)}
					onSubmitEditing={() => onNext(passwordRef)}
				/>
			</View>
			<View style={styles.inputView}>
				<TextInput
					ref={passwordRef}
					style={styles.TextInput}
					placeholder="Password"
					placeholderTextColor="#003f5c"
					secureTextEntry={true}
					autoCapitalize={"none"}
					onChangeText={(text) => setValue("password", text)}
					onSubmitEditing={handleSubmit(onValid)}
				/>
			</View>
			<TouchableOpacity onPress={() => navigation.navigate("Sign")}>
				<Text style={styles.signin_button}>Sign IN</Text>
			</TouchableOpacity>
			<TouchableOpacity style={styles.loginBtn} onPress={handleSubmit(onValid)}>
				<Text>LOGIN</Text>
			</TouchableOpacity>
		</View>
	);
}

const styles = StyleSheet.create({
	container: {
		flex: 1,
		backgroundColor: "#fff",
		alignItems: "center",
		justifyContent: "center",
	},
	image: {
		width: 100,
		height: 100,
		resizeMode: "cover",
		marginBottom: 40,
	},
	inputView: {
		backgroundColor: "#FFC0CB",
		borderRadius: 30,
		width: "70%",
		height: 45,
		marginBottom: 20,
		alignItems: "center",
	},
	TextInput: {
		height: 50,
		flex: 1,
		padding: 10,
		marginLeft: 20,
	},
	signin_button: {
		height: 30,
		marginBottom: 10,
	},
	loginBtn: {
		width: "80%",
		borderRadius: 25,
		height: 50,
		alignItems: "center",
		justifyContent: "center",
		marginTop: 40,
		backgroundColor: "#F48FB1",
	},
});
