import Layout from '../../components/layout';
import { search } from '../../lib/search';
import Head from 'next/head';
import utilStyles from '../../styles/utils.module.css';

export default function Search({ searchResult }) {
  return (
    <Layout>
      <Head>
        <title>REsult</title>
      </Head>
      <article>
        <h1 className={utilStyles.headingXl}>REsult</h1>
        <div className={utilStyles.lightText}>
          {searchResult.map(({ score, payload: { id, searchable_content} }) => (
            <div key={`id_${id}`}>
              <p>{score}</p>
              <p>{searchable_content}</p>
            </div>
            
          ))}
        </div>
      </article>
    </Layout>
  );
}

export async function getServerSideProps({ query }) {
  const { s } = query;
  const searchResult = await search(s);
  return {
    props: {
      searchResult,
    },
  };
}
