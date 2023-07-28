import AppNavigation from "./StackNavigator";
import { ImageBackground, Image } from "react-native";
import { useEffect, useState } from "react";

export default function App() {
  const [isLoading, setIsLoading] = useState(false); //true로 바꾸면 splash 가능
  useEffect(() => {
    // 1,000ms가 1초
    setTimeout(() => {
      setIsLoading(false);
    }, 2000);
  }, []);

  const Loading = () => {
    return (
      <Image
        source={require("./assets/wineSplash2.jpg")}
        style={{ width: "100%", height: "100%" }}
      ></Image>
    );
  };

  if (isLoading) {
    return <Loading />;
  } else {
    return <AppNavigation />;
  }
}
