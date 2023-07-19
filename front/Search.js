import React, { useState } from "react";
import {
    StyleSheet,
    Text,
    View,
    Image,
    TextInput,
    TouchableOpacity,
    FlatList,
    SafeAreaView
  } from "react-native";
import Icon from 'react-native-vector-icons/Ionicons';
import wine from "./wineList";

export default function Search({navigation}) {
  const [search, setSearch] = useState("");

  const isWinePress = (item) => { // 와인 눌렀을 때 상세페이지로 넘어가도록
    console.log(item.name)
    navigation.navigate('WineInfo', {id: item.wineId})
  }

  const renderItem = ({ item }) => {
    if (item.name.includes(search)) {
      return(
        <TouchableOpacity 
          onPress={() => isWinePress(item)}
          style={styles.content}
        >
          <Image style={styles.image} source={item.pic} />
          <View style={{padding: 15}}>
            <Text style={{fontSize: 25}}>{item.name}</Text>
            <View style={{flexDirection: "row", alignItems: 'center'}}>
              <Icon name={'star'} size={15} color={"#000000"}/>
              <Text style={{fontSize: 18, paddingLeft: 5}}>{item.rating}</Text>
            </View>
          </View>
        </TouchableOpacity>
      )
    }
  }

  return (
      <View style={styles.container}>
        <View style={styles.inputView}>
          <TextInput
            style={styles.searchBar}
            placeholder="검색어를 입력하세요"
            onChangeText={(search) => setSearch(search)}
          />
        </View>
        <SafeAreaView style={{marginBottom: 90, borderRadius: 20, backgroundColor: "#FFFFFF", height: "100%"}}>
          <FlatList
            data={wine}
            renderItem={renderItem}
            keyExtractor={(item) => item.wineId}
          >
          </FlatList>
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
  },
  searchBar: {
    padding: 10,
    height: 50,
    alignItems: "center",
    paddingLeft: 30,
  },
  inputView: {
    backgroundColor: "#E0E0E0",
    borderRadius: 20,
    //width: "90%",
    height: 50,
    marginBottom: 20,
    marginTop: 20,
    flexDirection: "row"
  },
  content: {
    paddingHorizontal: 10,
    borderRadius: 20,
    flexDirection: "row",
    margin: 7,
  },
})