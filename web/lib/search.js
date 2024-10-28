
export async function search(searchString) {
  const baseUrl = 'http://localhost:9000';
  const searchParams = new URLSearchParams({text: searchString});
  const url = new URL(`${baseUrl}/search?${searchParams.toString()}`);
  const res = await fetch(url, {
    method: 'GET',
  });

  return await res.json();
}
