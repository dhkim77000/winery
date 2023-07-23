import { useReactiveVar } from "@apollo/client";
import { set, useForm } from "react-hook-form";
import {
	Text, 
	View,
	Image,
	StyleSheet, 
	TouchableOpacity, 
	Dimensions, 
	SafeAreaView, 
	ScrollView, 
	FlatList, 
} from "react-native";
import { emailVar, postApi } from "./Api";
import { useEffect, useState } from "react";
import Icon from 'react-native-vector-icons/Ionicons';

export default function Recommend({navigation}) {
	const email = useReactiveVar(emailVar);
	const { register, handleSubmit, setValue, watch } = useForm();
	const [selectedId, setSelectedId] = useState(0); // category 선택
	const [wineList, setWineList] = useState([]);

	const categories = [
		{ id: 0, text: '인기순', icon: 'star' },
		{ id: 1, text: '추천순', icon: 'heart' },
		//{ id: 2, text: '여름추천', icon: 'sunny', item: [] },
	]

	const onValid = async (data) => {
		const endpoint = "wine/wine_recommend/";
		console.log(data)
		try {
			const response = await postApi(endpoint, data);
			setWineList(response.data)
		} catch (error) {
			alert(error);
			console.log(error);
		}
	};


	const CategoryBox = ({ isSelected, category }) => {
		return (
			<TouchableOpacity style={[
				styles.categoryBox,
				{opacity: isSelected ? 0.7 : 1}]} 
				onPress={() => {
					setSelectedId(category.id);
					setValue("type", category.text); 
					handleSubmit(onValid)();
				}
				}>
			{isSelected
					? <Icon name={category.icon} size={30} color={"#000000"} />
					: <Text style={styles.categoryText}>{category.text}</Text>}
			</TouchableOpacity>
		)
	}

	const isWinePress = (item) => { // 와인 눌렀을 때 상세페이지로 넘어가도록
		console.log(item.item_id)
		navigation.navigate('WineInfo', {item: item})
	}

	const renderItem = ({item}) => {

		return(
			<TouchableOpacity 
				onPress={() => isWinePress(item)}
				style={styles.content}
			>
			<Image style={styles.image} source={require('./assets/고라파덕.jpg')} />
			<View style={{padding: 15, flexShrink:1}}>
				<Text style={{fontSize: 18, marginBottom: 5}}>{item.name}</Text>
				<View style={{flexDirection: "row", alignItems: 'center'}}>
					<Icon name={'star'} size={15} color={"#000000"}/>
					<Text style={{fontSize: 18, paddingLeft: 5}}>{item.wine_rating}</Text>
				</View>
				<View style={{flexDirection: "row", alignItems: "center"}}> 
					<Text>{item.country},  {item.region1}</Text>
				</View>
				<View style={{flexDirection: "row", alignItems: "center", marginTop: 5}}> 
					<Text>$ {item.price}</Text>
				</View>
			</View>
			</TouchableOpacity>
		)
	}

	useEffect(() => {
		register("email", {
			required: true,
		});
		register("type", {
			required: true,
		});
		setValue("email", email);
		setValue("type", "인기순")
		handleSubmit(onValid)();
	}, [register]);

	return (
		<View style={styles.wrapper}>
		  <View style={styles.subHeader}>
			<ScrollView horizontal={true} showsHorizontalScrollIndicator={false}>
			  {categories.map((category, index) => {
				const isSelected = category.id === selectedId;
				return <CategoryBox key={index} isSelected={isSelected} category={category}/>
			  })}
			</ScrollView>
		  </View>
		  <SafeAreaView style={{marginBottom: 90, borderRadius: 20, backgroundColor: "#FFFFFF", height: "100%"}}>
			<FlatList
				data={Object.values(wineList)}
				renderItem={renderItem}
				keyExtractor={(item) => item.item_id}
			>
			</FlatList>
			<View style={{height: 80}}/>
		  </SafeAreaView>
		</View>
	);
}

const styles = StyleSheet.create({
	wrapper: {
	  flex: 1,
	  backgroundColor: "#FFC0CB",
	},
	subHeader: {
	  paddingHorizontal: 15,
	  paddingTop: 10,
	  // borderWidth: 1,
	  marginTop: 10,
	  marginBottom: 10,
	  justifyContent: 'center',
	},
	categoryView: {
	  marginBottom: 4,
	},
	categoryBox: {
	  backgroundColor: "#FFFFFF",
	  borderRadius: 14,
	  width: (Dimensions.get('window').width-32-30)/3.7,
	  paddingVertical: 6,
	  alignItems: 'center',
	  justifyContent: 'center',
	  marginBottom: 10,
	  marginRight: 10,
	  ...Platform.select({
		  ios: {
			shadowOpacity: 0.2,
			shadowRadius: 4,
			shadowOffset: {width: 3, height: 3},
		  },
		  android: {
			elevation: 6,
		  },
	  }),
	},
	categoryText: {
	  includeFontPadding: false,
	  color: "#000000",
	  fontSize: 15,
	},
	content: {
	  paddingHorizontal: 10,
	  borderRadius: 20,
	  flexDirection: "row",
	  margin: 7,
	},
	postButtonWrap: {
	  position: 'absolute',
	  margin: 16,
	  right: 0,
	  bottom: 0,
	},
	header: {
	  paddingVertical: 14,
	  paddingHorizontal: 16,
	  alignItems: 'center'
	  // borderWidth: 1,
	},
	title: {
		fontSize: 24,
		fontWeight: '700',
	},
	image: {
	  width: 90,
	  height: 90,
	  margin: 10,
	  //resizeMode: "contain",
	}
  })