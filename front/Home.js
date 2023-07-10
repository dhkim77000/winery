// import React, { Component } from 'react';
import {
    StyleSheet,
    Text,
    View,
    Image,
    TextInput,
    Button,
    TouchableOpacity,
  } from "react-native";
  
  export default function Home({navigation}) {
      return (
      <View style={styles.container}>
        <Text style={styles.paragraph}>
          Temporary Home
        </Text>
        <TouchableOpacity style={styles.tempbtn} onPress={() => navigation.navigate('Recommend')}>
          <Text>Recommend</Text> 
        </TouchableOpacity>
        <TouchableOpacity style={styles.tempbtn} onPress={() => navigation.navigate('Group')}>
          <Text>Group Recommend</Text> 
        </TouchableOpacity> 
      </View>
    );
  }
  
  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: "#fff",
      alignItems: "center",
      justifyContent: "center",
    },
    paragraph: {
      margin: 24,
      fontSize: 18,
      fontWeight: 'bold',
      textAlign: 'center',
    },
    item: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      height: 50,
      paddingHorizontal: 20,
      borderTopWidth: 1,
      borderColor: '#000',
    },
    tempbtn: {
      width: "80%",
      borderRadius: 25,
      height: 50,
      alignItems: "center",
      justifyContent: "center",
      marginTop: 20,
      marginBottom: 20,
      backgroundColor: "#ffc0cb",
    },
  })