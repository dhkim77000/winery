import { useReactiveVar } from "@apollo/client";
import { useForm } from "react-hook-form";
import {
	Text,
	View,
	StyleSheet,
	TouchableOpacity,
	ScrollView,
	FlatList,
} from "react-native";
import { emailVar, postApi } from "./Api";
import { useEffect } from "react";

export default function Recommend() {
	const email = useReactiveVar(emailVar);
	const { register, handleSubmit, setValue, watch } = useForm();
	const onValid = async (data) => {
		const endpoint = "wine/wine_recommend/";
		console.log(data)
		try {
			const response = await postApi(endpoint, data);
			console.log(response.data)
			// setValue("wine_list", response.data.wine_list);
		} catch (error) {
			alert(error);
			console.log(error);
		}
	};
	const renderItem = ({ item }) => (
		<Text style={{ marginVertical: 10 }} key={item.tier}>
			{item.tier}. {item.wine_name}
		</Text>
	);
	useEffect(() => {
		register("email", {
			required: true,
		});
		register("type", {
			required: true,
		});
		// register("wine_list", {
		// 	required: false,
		// });
		setValue("email", email);
	}, [register]);

	return (
		<View
			style={{
				flex: 1,
				backgroundColor: "#fff",
				alignItems: "center",
				justifyContent: "center",
			}}
		>
			<Text>This is Recommend page</Text>
			<TouchableOpacity
				style={{
					width: "80%",
					borderRadius: 25,
					height: 50,
					alignItems: "center",
					justifyContent: "center",
					marginTop: 40,
					backgroundColor: "#F48FB1",
				}}
				onPress={() => {
					setValue("type", "type1");
					handleSubmit(onValid)();
				}}
			>
				<Text>Ranking Type 1</Text>
			</TouchableOpacity>
			<TouchableOpacity
				style={{
					width: "80%",
					borderRadius: 25,
					height: 50,
					alignItems: "center",
					justifyContent: "center",
					marginTop: 40,
					backgroundColor: "#F48FB1",
				}}
				onPress={() => {
					setValue("type", "type2");
					handleSubmit(onValid)();
				}}
			>
				<Text>Ranking Type 2</Text>
			</TouchableOpacity>
			<View
				style={{
					width: "80%",
					borderRadius: 25,
					alignItems: "center",
					justifyContent: "center",
					marginTop: 40,
					backgroundColor: "#F48FB1",
				}}
			>
				{/* <FlatList
					data={watch("rank")}
					renderItem={renderItem}
					keyExtractor={(item) => item.tier}
				/> */}
			</View>
		</View>
	);
}
