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
	TouchableWithoutFeedback,
	Keyboard,
	KeyboardAvoidingView,
} from "react-native";
import { postApi, getApi, isLoggedInVar } from "./Api";

export default function Login({ navigation }) {
	const { register, handleSubmit, setValue, watch } = useForm();
	const passwordRef = useRef();

	const onNext = (nextOne) => {
		nextOne?.current?.focus();
	};
	const onValid = async (data) => {
		const regex = /\w+@\w+\.[\w,\.]+/;
		console.log(data);
		if (regex.test(data.email)) {
			try {
				const response = await postApi("temp/login/", data);
				console.log(response.data.status);
				if (response.data.status) {
					isLoggedInVar(true);
				} else {
					alert("회원정보를 찾을 수 없습니다");
				}
			} catch (error) {
				alert(error);
				console.log(error);
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
		<TouchableWithoutFeedback style={{ flex: 1 }} onPress={Keyboard.dismiss}>
			<KeyboardAvoidingView style={{ flex: 1 }} behavior="padding" enabled>
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
					<TouchableOpacity
						style={styles.loginBtn}
						onPress={handleSubmit(onValid)}
					>
						<Text>LOGIN</Text>
					</TouchableOpacity>
				</View>
			</KeyboardAvoidingView>
		</TouchableWithoutFeedback>
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
