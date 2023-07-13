import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";

import Home from "./Home";
import Login from "./Login";
import Sign from "./Sign";

import Icon from 'react-native-vector-icons/Ionicons';

import Recommend from "./Recommend";
import Group from "./Group";

import {
  StyleSheet,
  Text,
} from "react-native";

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

export default function AppNavigation() {
  return(
    <NavigationContainer>
      <Stack.Navigator 
        screenOptions={{
          contentStyle: {backgroundColor: 'white'}
        }}
        initialRouteName="Login"
      >
        <Stack.Screen
          name="Login"
          component={Login}
          //options={{ headerShown: false }}
        ></Stack.Screen>
        <Stack.Screen
          name="Sign"
          component={Sign}
          //options={{ headerShown: false }}
        ></Stack.Screen>
        <Stack.Screen
          name="TabNavi"
          component={TabNavi}
          options={{
            headerRight: () => (
              <Icon name="menu" color="#000000" size={22} style={{ paddingRight:10 }}/>
            ),
            headerTitle: 'WINERY',
          }}
          //options={{ headerShown: false }}
        ></Stack.Screen>
      </Stack.Navigator>
    </NavigationContainer>
  )
}

function TabNavi() {
  return <Tab.Navigator 
      initialRouteName="Home"
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,
      }}
    >
      <Tab.Screen
        name="Recommend"
        component={Recommend}
        options={{
          title: '추천 페이지',
          tabBarIcon: () => (
            <Icon name="heart" color="#000000" size={24} />
          ),
        }}
      />
      <Tab.Screen
        name="Home"
        component={Home}
        options={{
          title: '메인 페이지',
          tabBarIcon: () => (
            <Icon name="home" color="#000000" size={24} />
          ),
        }}
      />
      <Tab.Screen
        name="Group"
        component={Group}
        options={{
          title: '그룹 추천 페이지',
          tabBarIcon: () => (
            <Icon name="people" color="#000000" size={24} />
          ),
        }}
      />
    </Tab.Navigator>
}

