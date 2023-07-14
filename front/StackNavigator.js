import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
// import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";

import Home from "./Home";
import Login from "./Login";
import Sign from "./Sign";

// import Icon from 'react-native-vector-icons/MaterialIcons';

import Recommend from "./Recommend";
import Group from "./Group";

// const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

export default function AppNavigation() {
  return(
    <NavigationContainer>
      <WholeStack />
    </NavigationContainer>
  )
}

function WholeStack() {
  return (
    <Stack.Navigator 
      screenOptions={{
        contentStyle: {backgroundColor: 'white'},
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
        //options={{ headerShown: false }}
      />
      <Stack.Screen
        name="Sign"
        component={Sign}
        //options={{ headerShown: false }}
      />
      <Stack.Screen
        name="Home"
        component={Home}
        //options={{ headerShown: false }}
      />
      <Stack.Screen
        name="Recommend"
        component={Recommend}
        //options={{ headerShown: false }}
      />
      <Stack.Screen
        name="Group"
        component={Group}
        //options={{ headerShown: false }}
      />
    </Stack.Navigator>
  )
}

/*
<Stack.Screen
        name="TabNavi"
        options={{ headerShown: false }}
      >
        {() => (
          <TabNavi />
        )}
      </Stack.Screen>
      
function TabNavi() {
  return(
    <Tab.Navigator 
      initialRouteName="Home"
      screenOptions={({route}) => ({
        headerShown: false,
        tabBarShowLabel: false,
        tabBarStyle: {

        }
      })}
    >
      <Tab.Screen
        name="Recommend"
        component={Recommend}
        options={{
          title: '추천 페이지',
          tabBarIcon: () => (
            <Icon name="heart" color={"white"} size={24} />
          ),
        }}
      />
      <Tab.Screen
        name="Home"
        component={Home}
        options={{
          title: '메인 페이지',
          tabBarIcon: () => (
            <Icon name="home" color={"white"} size={24} />
          ),
        }}
      />
      <Tab.Screen
        name="Group"
        component={Group}
        options={{
          title: '그룹 추천 페이지',
          tabBarIcon: () => (
            <Icon name="people" color={"white"} size={24} />
          ),
        }}
      />
    </Tab.Navigator>
  )
}
*/

