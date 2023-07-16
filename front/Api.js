import { makeVar } from "@apollo/client";
import axios from "axios";

export const isLoggedInVar = makeVar(false);
export const emailVar = makeVar("");

export const logUserIn = async (email) => {
	isLoggedInVar(true);
	emailVar(email);
};

export const logUserOut = async () => {
	isLoggedInVar(false);
	emailVar(null);
};

const api = axios.create({
	baseURL: "http://34.16.153.40:8000/", // FastAPI 백엔드의 기본 URL을 여기에 입력합니다.
});

export const getApi = (endpoint) => {
	// endpoint: login/, login/register/, home/ ...
	return api.get(endpoint); // 백엔드에서 제공하는 엔드포인트로 GET 요청을 보냅니다.
};

export const postApi = async (endpoint, data) => {
	// endpoint: login/, login/register/, home/ ...
	return await api.post(endpoint, data); // 엔드포인트로 POST 요청을 보냅니다.
};
