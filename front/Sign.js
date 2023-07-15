import { StatusBar } from "expo-status-bar";
import React, { useState, Component, useRef, useEffect } from "react";
import { useForm } from "react-hook-form";
import {
	StyleSheet,
	Text,
	View,
	Image,
	TextInput,
	Button,
	TouchableOpacity,
	Keyboard,
	TouchableWithoutFeedback,
	KeyboardAvoidingView,
} from "react-native";
import { isLoggedInVar, postApi } from "./Api";

export default function Sign() {
	const { register, handleSubmit, setValue, watch } = useForm();
	const passwordRef = useRef();
	const passwordCheckRef = useRef();

	const onNext = (nextOne) => {
		nextOne?.current?.focus();
	};
	const onValid = async (data) => {
		const endpoint = "temp/signin/";
		const regex = /\w+@\w+\.[\w,\.]+/;
		if (!regex.test(data.email)) {
			alert("이메일 형식이 맞지 않습니다");
		} else if (data.password != data.password_check) {
			alert("비밀번호가 일치하지 않습니다");
		} else {
			delete data.password_check; // password_check 항목 제거
			try {
				const response = await postApi(endpoint, data);
				console.log(response.data)
				if (!response.data.status) {
					alert("이미 존재하는 계정입니다")
				}
			} catch (error) {
				alert(error);
				console.log(error);
			}
		}
	};

	useEffect(() => {
		register("email", {
			required: true,
		});
		register("password", {
			required: true,
		});
		register("password_check", {
			required: true,
		});
	}, [register]);

	return (
		<TouchableWithoutFeedback style={{ flex: 1 }} onPress={Keyboard.dismiss}>
			<KeyboardAvoidingView style={{ flex: 1 }} behavior="padding" enabled>
				<View style={styles.container}>
					<StatusBar style="auto" />
					<View style={styles.inputView}>
						<TextInput
							style={styles.TextInput}
							placeholder="Email"
							placeholderTextColor="#003f5c"
							onChangeText={(text) => setValue("email", text)}
							onSubmitEditing={() => onNext(passwordRef)}
							autoCapitalize={"none"}
						/>
					</View>
					<View style={styles.inputView}>
						<TextInput
							style={styles.TextInput}
							placeholder="Password"
							placeholderTextColor="#003f5c"
							secureTextEntry={true}
							ref={passwordRef}
							onChangeText={(text) => setValue("password", text)}
							onSubmitEditing={() => onNext(passwordCheckRef)}
							autoCapitalize={"none"}
						/>
					</View>
					<View style={styles.inputView}>
						<TextInput
							style={styles.TextInput}
							placeholder="Check Password"
							placeholderTextColor="#003f5c"
							secureTextEntry={true}
							ref={passwordCheckRef}
							onChangeText={(text) => setValue("password_check", text)}
							onSubmitEditing={handleSubmit(onValid)}
							autoCapitalize={"none"}
						/>
					</View>
					<TouchableOpacity
						style={styles.nextBtn}
						onPress={handleSubmit(onValid)}
					>
						<Text>NEXT</Text>
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
	nextBtn: {
		width: "80%",
		borderRadius: 25,
		height: 50,
		alignItems: "center",
		justifyContent: "center",
		marginTop: 40,
		backgroundColor: "#F48FB1",
	},
});
