import React from "react";
import { Image, StyleSheet, TouchableOpacity, Platform } from "react-native";
import PropTypes from "prop-types";
import { createIconSetFromIcoMoon } from "react-native-vector-icons";
import EntypoIcons from "react-native-vector-icons/Entypo";
import EvilIconsIcons from "react-native-vector-icons/EvilIcons";
import FeatherIcons from "react-native-vector-icons/Feather";
import FontAwesomeIcons from "react-native-vector-icons/FontAwesome";
import FoundationIcons from "react-native-vector-icons/Foundation";
import IoniconsIcons from "react-native-vector-icons/Ionicons";
import MaterialIconsIcons from "react-native-vector-icons/MaterialIcons";
import MaterialCommunityIconsIcons from "react-native-vector-icons/MaterialCommunityIcons";
import OcticonsIcons from "react-native-vector-icons/Octicons";
import ZocialIcons from "react-native-vector-icons/Zocial";
import SimpleLineIconsIcons from "react-native-vector-icons/SimpleLineIcons";

const iconSets = {
  Entypo: EntypoIcons,
  EvilIcons: EvilIconsIcons,
  Feather: FeatherIcons,
  FontAwesome: FontAwesomeIcons,
  Foundation: FoundationIcons,
  Ionicons: IoniconsIcons,
  MaterialIcons: MaterialIconsIcons,
  MaterialCommunityIcons: MaterialCommunityIconsIcons,
  Octicons: OcticonsIcons,
  Zocial: ZocialIcons,
  SimpleLineIcons: SimpleLineIconsIcons,
};

const propTypes = {
  buttonStyle: PropTypes.any,
  disabled: PropTypes.bool,
  halfStarEnabled: PropTypes.bool,
  icoMoonJson: PropTypes.object,
  iconSet: PropTypes.string,
  rating: PropTypes.number,
  reversed: PropTypes.bool,
  starColor: PropTypes.string,
  starIconName: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number,
    PropTypes.object,
  ]),
  starSize: PropTypes.number,
  activeOpacity: PropTypes.number,
  starStyle: PropTypes.any,
  onStarButtonPress: PropTypes.func,
};

const defaultProps = {
  buttonStyle: {},
  disabled: false,
  halfStarEnabled: true,
  icoMoonJson: undefined,
  iconSet: "FontAwesome",
  rating: 0,
  reversed: false,
  starColor: "black",
  starIconName: "star",
  starSize: 40,
  activeOpacity: 0.2,
  starStyle: {},
  onStarButtonPress: () => {},
};

const StarButton = ({
  buttonStyle,
  disabled,
  halfStarEnabled,
  icoMoonJson,
  iconSet,
  rating,
  reversed,
  starColor,
  starIconName,
  starSize,
  activeOpacity,
  starStyle,
  onStarButtonPress,
}) => {
  const onButtonPress = (event) => {
    let addition = 0;

    if (halfStarEnabled) {
      const touchX =
        Platform.OS === "web"
          ? event.nativeEvent.pageX -
            event.currentTarget.getBoundingClientRect().left
          : event.nativeEvent.locationX; // Platform 별 처리
      const starWidth = starSize;
      const isHalfSelected = touchX < starWidth / 2;
      addition = isHalfSelected ? -0.5 : 0;
    }

    onStarButtonPress(rating + addition);
  };

  const iconSetFromProps = () => {
    if (icoMoonJson) {
      return createIconSetFromIcoMoon(icoMoonJson);
    }

    return iconSets[iconSet];
  };

  const renderIcon = () => {
    const Icon = iconSetFromProps();
    let iconElement;

    const newStarStyle = {
      transform: [
        {
          scaleX: reversed ? -1 : 1,
        },
      ],
      ...StyleSheet.flatten(starStyle),
    };

    if (typeof starIconName === "string") {
      iconElement = (
        <Icon
          name={starIconName}
          size={starSize}
          color={starColor}
          style={newStarStyle}
        />
      );
    } else if (typeof starIconName === "number") {
      iconElement = (
        <Image
          source={starIconName}
          style={{ width: starSize, height: starSize, ...newStarStyle }}
        />
      );
    } else {
      iconElement = null;
    }

    return iconElement;
  };

  return (
    <TouchableOpacity
      activeOpacity={activeOpacity}
      disabled={disabled}
      style={buttonStyle}
      onPress={onButtonPress}
    >
      {renderIcon()}
    </TouchableOpacity>
  );
};

StarButton.propTypes = propTypes;
StarButton.defaultProps = defaultProps;

export default StarButton;
