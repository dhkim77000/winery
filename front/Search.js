import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import {
  StyleSheet,
  Text,
  View,
  Image,
  TextInput,
  TouchableOpacity,
  FlatList,
  SafeAreaView,
} from "react-native";

import { postApi } from "./Api";
import Icon from "react-native-vector-icons/Ionicons";
import picData from "./mapped_idx2item.json";

export default function Search({ navigation }) {
  const [searchText, setSearchText] = useState("");
  const { register, handleSubmit, setValue, watch } = useForm();
  const [wineList, setWineList] = useState([]);

  const onValid = async (data) => {
    const endpoint = "wine/search_by_name/";
    console.log(data);
    try {
      const response = await postApi(endpoint, data);
      setWineList(response.data);
      console.log(response.data);
    } catch (error) {
      alert(error);
      console.log(error);
    }
  };

  useEffect(() => {
    register("name", {
      required: true,
    });
  }, [register]);

  const isWinePress = (item) => {
    // 와인 눌렀을 때 상세페이지로 넘어가도록
    console.log(item.item_id);
    navigation.navigate("WineInfo", { item: item });
  };

  const renderItem = ({ item }) => {
    return (
      <TouchableOpacity
        onPress={() => isWinePress(item)}
        style={styles.content}
      >
        <Image
          style={styles.image}
          source={
            picData[item.item_id]
              ? { uri: picData[item.item_id] }
              : require("./assets/wine.jpg")
          }
        />
        <View style={{ padding: 15, flexShrink: 1 }}>
          <Text style={{ fontSize: 18, marginBottom: 5 }}>{item.name}</Text>
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <Icon name={"star"} size={15} color={"#000000"} />
            <Text style={{ fontSize: 18, paddingLeft: 5 }}>
              {item.wine_rating}
            </Text>
          </View>
          <View style={{ flexDirection: "row", alignItems: "center" }}>
            <Text>
              {item.country}, {item.region1}
            </Text>
          </View>
          <View
            style={{ flexDirection: "row", alignItems: "center", marginTop: 5 }}
          >
            <Text>$ {item.price}</Text>
          </View>
        </View>
      </TouchableOpacity>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.inputView}>
        <TextInput
          style={styles.searchBar}
          placeholder="검색어를 입력하세요"
          onChangeText={(search) => {
            setSearchText(search);
          }}
          onSubmitEditing={() => {
            setValue("name", searchText);
            onValid(watch());
          }}
        />
      </View>
      <SafeAreaView
        style={{
          marginBottom: 90,
          borderRadius: 20,
          backgroundColor: "#FFFFFF",
          height: "100%",
        }}
      >
        <FlatList
          data={Object.values(wineList)}
          renderItem={renderItem}
          keyExtractor={(item) => item.item_id}
        ></FlatList>
        <View style={{ height: 80 }} />
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  image: {
    width: 90,
    height: 90,
    margin: 10,
    resizeMode: "contain",
  },
  searchBar: {
    height: 50,
    flex: 1,
    padding: 10,
    marginLeft: 20,
  },
  inputView: {
    backgroundColor: "#E0E0E0",
    borderRadius: 20,
    //width: "90%",
    height: 50,
    marginBottom: 20,
    marginTop: 20,
    flexDirection: "row",
  },
  content: {
    paddingHorizontal: 10,
    borderRadius: 20,
    flexDirection: "row",
    margin: 7,
  },
});
