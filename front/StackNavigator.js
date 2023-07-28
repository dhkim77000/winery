import { NavigationContainer, useNavigation } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";

import Icon from "react-native-vector-icons/Ionicons";
import { Platform, TouchableOpacity, View, Text } from "react-native";

import Home from "./Home";
import Login from "./Login";
import Sign from "./Sign";
import Mbti from "./Mbti";
import Recommend from "./Recommend";
import Search from "./Search";
import WineInfo from "./WineInfo";

import { useReactiveVar } from "@apollo/client";

import { isLoggedInVar } from "./Api";

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();
const CustomHeader = ({ title }) => {
  const navigation = useNavigation();

  return (
    <View
      style={{
        width: 500, // 원하는 너비로 설정
        backgroundColor: "#FFC0CB",
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        paddingHorizontal: 16,
        height: 50,
        alignSelf: "center",
        //elevation: 5, // Android에서 그림자 효과 적용
        shadowColor: "black", // iOS에서 그림자 색상
        shadowOffset: { width: 0, height: 2 }, // iOS에서 그림자의 좌표 이동
        shadowOpacity: 0.3, // iOS에서 그림자 투명도
        shadowRadius: 5, // iOS에서 그림자의 블러 반경
      }}
    >
      <TouchableOpacity onPress={() => navigation.goBack()}>
        {title != "WINERY" ? (
          <Icon
            name={"arrow-back-circle-outline"}
            size={30}
            color={"#FFFFFF"}
          />
        ) : (
          <View style={{ width: 30 }} />
        )}
      </TouchableOpacity>
      <Text style={{ fontSize: 20, fontWeight: "600", color: "#FFFFFF" }}>
        {title}
      </Text>
      <View style={{ width: 30 }} />
    </View>
  );
};

export default function AppNavigation() {
  const isLoggedIn = useReactiveVar(isLoggedInVar);
  return (
    <NavigationContainer>
      {isLoggedIn ? <LoggedInNav /> : <LoggedOutNav />}
    </NavigationContainer>
  );
}

function LoggedInNav() {
  return (
    <Stack.Navigator
      screenOptions={{
        contentStyle: {
          backgroundColor: "white",
          ...Platform.select({
            web: {
              width: 500,
              alignSelf: "center",
            },
          }),
        },
        headerStyle: {
          backgroundColor: "#FFC0CB",
          height: 50,
        },
        headerTintColor: "#FFFFFF",
      }}
    >
      <Stack.Screen
        name="TabNavi"
        component={TabNavi}
        options={() => ({
          headerTitle: "WINERY",
          headerShown: false,
        })}
      />
      <Stack.Screen
        name="WineInfo"
        component={WineInfo}
        options={{
          ...Platform.select({
            web: {
              header: () => <CustomHeader title=" " />,
            },
          }),
        }}
      />
    </Stack.Navigator>
  );
}

function LoggedOutNav() {
  return (
    <Stack.Navigator
      screenOptions={{
        contentStyle: {
          backgroundColor: "white",
          ...Platform.select({
            web: {
              width: 500,
              alignSelf: "center",
            },
          }),
        },
        headerStyle: {
          backgroundColor: "#FFC0CB",
          height: 50,
        },
        headerTintColor: "#FFFFFF",
      }}
      initialRouteName="Login"
    >
      <Stack.Screen
        name="Login"
        component={Login}
        options={{
          ...Platform.select({
            web: {
              header: () => <CustomHeader title="WINERY" />,
            },
            android: {
              headerTitle: "WINERY",
            },
            ios: {
              headerTitle: "WINERY",
            },
          }),
        }}
      />
      <Stack.Screen
        name="Sign"
        component={Sign}
        options={{
          ...Platform.select({
            web: {
              header: () => <CustomHeader title="Sign" />,
            },
          }),
        }}
      />
      <Stack.Screen
        name="Mbti"
        component={Mbti}
        options={{
          headerStyle: {
            backgroundColor: "#FFC0CB",
          },
          ...Platform.select({
            web: {
              header: () => <CustomHeader title="Find your WINE STYLE!" />,
            },
            ios: {
              headerTitle: "Find your WINE STYLE!",
            },
            android: {
              headerTitle: "Find your WINE STYLE!",
            },
          }),
        }}
      />
    </Stack.Navigator>
  );
}

function TabNavi() {
  return (
    <Tab.Navigator
      initialRouteName="Home"
      screenOptions={{
        tabBarShowLabel: false,
        tabBarStyle: { height: 100 },
        tabBarActiveBackgroundColor: "#FFDDE3",
        //headerTintColor: "#FFFFFF",
      }}
    >
      <Tab.Screen
        name="Recommend"
        component={Recommend}
        options={{
          title: "추천 페이지",
          tabBarIcon: () => <Icon name="heart" color="#000000" size={24} />,
        }}
      />
      <Tab.Screen
        name="Home"
        component={Home}
        options={{
          title: "WELCOME TO WINERY",
          tabBarIcon: () => <Icon name="home" color="#000000" size={24} />,
        }}
      />
      <Tab.Screen
        name="Search"
        component={Search}
        options={{
          title: "검색 페이지",
          tabBarIcon: () => <Icon name="search" color="#000000" size={24} />,
        }}
      />
    </Tab.Navigator>
  );
}
