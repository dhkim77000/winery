import React, { useEffect, useState } from "react";
import { StyleSheet, Text, View, TouchableOpacity } from "react-native";
import questions from "./MbtiQ";
import { useForm } from "react-hook-form";
import { postApi } from "./Api";

export default function Mbti({ navigation: { navigate, goBack }, route }) {
	console.log(route.params.email);
	console.log(route.params.password);
	const { register, setValue, watch } = useForm();
	const [questionNum, setQuestionNum] = useState(0); // 문항번호 0~9
	const [result, setResult] = useState([]); // 고른 결과 리스트

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
		setValue("email", route.params.email);
		setValue("password", route.params.password);
	}, []);

	const isPress = (item) => {
		// 선택지 눌렀을 때
		setResult([...result, item.label]);
		setQuestionNum(questionNum + 1);
	};

	const onValid = async (data) => {
		const endpoint = "login/register/";
		console.log(data);
		try {
			const response = await postApi(endpoint, data);
			console.log(response.data);
		} catch (error) {
			alert(error);
			console.log(error);
		}
	};

	const isBack = () => {
		// back 눌렀을 때 전 문제로 돌아가기
		setQuestionNum(questionNum - 1);
		result.pop();
		setResult([...result]);
	};

	console.log(result); // 결과 잘 나오는지 실험용

	if (questionNum === 10) {
		//완료 페이지
		return (
			<View style={styles.wrapper}>
				<Text style={{ fontSize: 25, marginBottom: 20 }}>결과</Text>
				<Text>{result}</Text>
				<TouchableOpacity
					style={styles.doneBtn}
					onPress={() => {
						setValue("mbti_result", result);
						onValid(watch());
						navigate("Login");
					}}
				>
					<Text>처음으로 돌아가기</Text>
				</TouchableOpacity>
			</View>
		);
	} else if ((questionNum < 10) & (questionNum > -1)) {
		return (
			<View style={styles.wrapper}>
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
	} else {
		// 전 sign 페이지로 back
		goBack();
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
		width: "60%",
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
		padding: 10,
	},
	backBtn: {
		width: "30%",
		borderRadius: 25,
		height: 50,
		alignItems: "center",
		justifyContent: "center",
		backgroundColor: "#F48FB1",
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
	},
});
