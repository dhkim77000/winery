import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";

import Icon from 'react-native-vector-icons/Ionicons';
import { TouchableOpacity } from "react-native";

import Home from "./Home";
import Login from "./Login";
import Sign from "./Sign";
import Mbti from './Mbti';
import Recommend from "./Recommend";
import Search from './Search';
import WineInfo from './WineInfo';

import { useReactiveVar } from "@apollo/client";

import { isLoggedInVar } from "./Api";

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

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
				contentStyle: { backgroundColor: "white" },
				//headerBackTitleVisible: false,
				//headerTitle: false,
				//headerTransparent: true,
				//headerTintColor: "white",
			}}
			// initialRouteName="Login"
		>
		<Stack.Screen
          name="TabNavi"
          component={TabNavi}
          options={() => ({
            headerTitle: 'WINERY',
            headerShown: false
          })
          }
    />
    <Stack.Screen
          name="WineInfo"
          component={WineInfo}
          options={{
            //headerShown: false,
            headerStyle: {
              backgroundColor: "#FFC0CB"
            }
          }}
    />
		</Stack.Navigator>
	);
}

function LoggedOutNav() {
	return (
		<Stack.Navigator
			screenOptions={{
				contentStyle: { backgroundColor: "white" },
				headerBackTitleVisible: false,
				headerTitle: false,
				headerTransparent: true,
				headerTintColor: "white",
			}}
			initialRouteName="Login"
		>
			<Stack.Screen
				name="Login"
				component={Login}
			/>
			<Stack.Screen
				name="Sign"
				component={Sign}
			/>
      <Stack.Screen
          name="Mbti"
          component={Mbti}
          options={{
            headerStyle: {
              backgroundColor: "#FFC0CB"
            }
          }}
        />
		</Stack.Navigator>
	);
}

function TabNavi() {
  return(
    <Tab.Navigator 
      initialRouteName="Home"
      screenOptions={{
        tabBarShowLabel: false,
        tabBarStyle: { height: 100,},
        tabBarActiveBackgroundColor: "#FFDDE3",
        headerRight: () => (
          <TouchableOpacity>
            <Icon 
              name="menu" 
              color="#000000" 
              size={22} 
              style={{ paddingRight:10 }}
            />
          </TouchableOpacity>
        ),
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
        name="Search"
        component={Search}
        options={{
          title: '검색 페이지',
          tabBarIcon: () => (
            <Icon name="search" color="#000000" size={24} />
          ),
        }}
      />
    </Tab.Navigator>
  )
}

